// Walks through Okapi's storage lifecycle on real SIFT vectors:
// ingest with seals -> MANIFEST updates -> restart from MANIFEST -> fan-out search,
// checked against exact search. From the repo root:
//
//	go run ./run
//	go run ./run -seal 1000      (more, smaller segments)
//
// The manifest code here is a sketch of what goes into storage/ on Thursday.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"okapi/dataset"
	"okapi/index/flat"
	"okapi/index/hnsw"
	"okapi/storage/segment"
)

const k, ef = 10, 50

var cfg = hnsw.Config{M: 16, EfConstruction: 100, Seed: 1}

// MANIFEST: the only file that says what's live. Anything on disk it doesn't list is garbage.
type manifest struct {
	Version  int      `json:"version"`   // bumped on every write, just to see the edits
	NextFile int      `json:"next_file"` // one counter for WALs and segments, like LevelDB
	Sealed   []string `json:"sealed"`
	WAL      string   `json:"wal"`
}

func main() {
	dataPath := flag.String("data", "data/sift/sift_query.fvecs", "fvecs file: first n vectors are inserted, the next 100 are queries")
	n := flag.Int("n", 9900, "vectors to insert")
	sealAt := flag.Int("seal", 3000, "seal the mutable segment when it holds this many vectors")
	out := flag.String("out", "run/okapi-data", "data folder (wiped at start)")
	flag.Parse()

	vecs, err := dataset.ReadFVecs(*dataPath)
	check(err)
	if len(vecs) < *n+100 {
		log.Fatalf("%s has %d vectors, need %d", *dataPath, len(vecs), *n+100)
	}
	base, queries := vecs[:*n], vecs[*n:*n+100]
	dim := len(base[0])

	check(os.RemoveAll(*out))
	db := &db{dir: *out}
	db.ingest(base, dim, *sealAt)
	printTree(*out)

	fmt.Println("\n== restart: everything comes back from MANIFEST ==")
	sealed, mutable := db.recover(dim)
	defer mutable.Close()
	search(sealed, mutable, base, queries)
}

type db struct {
	dir string
	m   manifest
}

func (d *db) path(parts ...string) string { return filepath.Join(append([]string{d.dir}, parts...)...) }

func (d *db) nextName(prefix, ext string) string {
	d.m.NextFile++
	return fmt.Sprintf("%s-%06d%s", prefix, d.m.NextFile, ext)
}

func (d *db) ingest(base [][]float32, dim, sealAt int) {
	check(os.MkdirAll(d.path("wal"), 0755))
	check(os.MkdirAll(d.path("segments"), 0755))

	// the WAL is listed before any write goes into it, or recovery wouldn't know to replay it
	d.m.WAL = d.nextName("wal", ".log")
	d.writeManifest("first WAL")
	seg, err := segment.Open(d.path("wal", d.m.WAL), dim, cfg)
	check(err)

	for i, v := range base {
		check(seg.Upsert(uint64(i), v))
		if seg.Len() >= sealAt {
			d.seal(seg)
		}
	}
	fmt.Printf("[stop]     %d vectors left in the mutable segment, only in %s\n", seg.Len(), d.m.WAL)
	check(seg.Close())
}

// seal: save -> new WAL -> fsync dirs -> MANIFEST -> delete old WAL. Each step only
// starts once the previous one is durable, so a crash anywhere leaves a state
// recovery can read: either the old MANIFEST (old WAL still there) or the new one.
func (d *db) seal(seg *segment.Segment) {
	n := seg.Len()
	segName := d.nextName("seg", "")
	walName := d.nextName("wal", ".log")

	oldWAL, err := seg.Seal(d.path("segments", segName), d.path("wal", walName))
	check(err)
	// new directory entries aren't durable until their parent dir is fsynced
	syncDir(d.path("segments", segName))
	syncDir(d.path("segments"))
	syncDir(d.path("wal"))

	d.m.Sealed = append(d.m.Sealed, segName)
	d.m.WAL = walName
	d.writeManifest("seal " + segName)

	// safe now: MANIFEST no longer points at it
	check(os.Remove(oldWAL))
	fmt.Printf("[seal]     %d vectors -> %s, live WAL now %s, deleted %s\n", n, segName, walName, filepath.Base(oldWAL))
}

// temp file -> fsync -> rename over MANIFEST -> fsync dir. rename is atomic, so a
// reader sees the old MANIFEST or the new one, never half of one.
func (d *db) writeManifest(why string) {
	d.m.Version++
	b, err := json.MarshalIndent(d.m, "", "  ")
	check(err)

	tmp := d.path("MANIFEST.tmp")
	f, err := os.Create(tmp)
	check(err)
	_, err = f.Write(b)
	check(err)
	check(f.Sync())
	check(f.Close())
	check(os.Rename(tmp, d.path("MANIFEST")))
	syncDir(d.dir)
	fmt.Printf("[manifest] v%d (%s): sealed=%v wal=%s\n", d.m.Version, why, d.m.Sealed, d.m.WAL)
}

func (d *db) recover(dim int) ([]*hnsw.Index, *segment.Segment) {
	b, err := os.ReadFile(d.path("MANIFEST"))
	check(err)
	var m manifest
	check(json.Unmarshal(b, &m))

	var sealed []*hnsw.Index
	for _, name := range m.Sealed {
		h, err := hnsw.Load(d.path("segments", name))
		check(err)
		fmt.Printf("[load]     %s: %d vectors\n", name, h.Len())
		sealed = append(sealed, h)
	}
	mutable, err := segment.Open(d.path("wal", m.WAL), dim, cfg)
	check(err)
	fmt.Printf("[replay]   %s: %d vectors back in the mutable segment\n", m.WAL, mutable.Len())
	return sealed, mutable
}

// fan-out: every segment can hold a true neighbour, so search all of them and keep the best k
func search(sealed []*hnsw.Index, mutable *segment.Segment, base, queries [][]float32) {
	exact := flat.New(len(base[0]))
	for i, v := range base {
		check(exact.Add(uint64(i), v))
	}

	hits := 0
	for _, q := range queries {
		var all []hnsw.Result
		for _, h := range sealed {
			r, err := h.Search(q, k, ef)
			check(err)
			all = append(all, r...)
		}
		r, err := mutable.Search(q, k, ef)
		check(err)
		all = append(all, r...)
		slices.SortFunc(all, func(a, b hnsw.Result) int {
			switch {
			case a.Dist < b.Dist:
				return -1
			case a.Dist > b.Dist:
				return 1
			}
			return 0
		})
		all = all[:min(k, len(all))]

		truth, err := exact.Search(q, k)
		check(err)
		for _, r := range all {
			if slices.ContainsFunc(truth, func(t flat.Result) bool { return t.ID == r.ID }) {
				hits++
			}
		}
	}
	fmt.Printf("[search]   %d queries over %d sealed + 1 mutable segment: recall@%d = %.4f\n",
		len(queries), len(sealed), k, float64(hits)/float64(len(queries)*k))
}

func printTree(root string) {
	fmt.Printf("\n== %s ==\n", root)
	check(filepath.WalkDir(root, func(p string, e fs.DirEntry, err error) error {
		if err != nil || p == root {
			return err
		}
		rel, _ := filepath.Rel(root, p)
		indent := strings.Repeat("    ", strings.Count(rel, string(filepath.Separator)))
		if e.IsDir() {
			fmt.Printf("%s%s/\n", indent, e.Name())
			return nil
		}
		info, err := e.Info()
		if err != nil {
			return err
		}
		fmt.Printf("%s%-16s %9d B\n", indent, e.Name(), info.Size())
		return nil
	}))
}

func syncDir(dir string) {
	f, err := os.Open(dir)
	check(err)
	check(f.Sync())
	check(f.Close())
}

func check(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
