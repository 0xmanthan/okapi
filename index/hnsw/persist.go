package hnsw

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"okapi/internal/errs"
)

// On disk, little-endian:
//
//	vectors.bin  n*dim float32, node i starts at byte i*dim*4 (no header, so it can be mmapped)
//	ids.bin      n u64
//	graph.bin    [dim][m][efConstruction][n][entry][maxLevel] as u32, then per node:
//	             [level], and for each layer 0..level: [count][count neighbour ids]
const (
	vectorsFile = "vectors.bin"
	idsFile     = "ids.bin"
	graphFile   = "graph.bin"
)

// way above anything real: P(level >= l) = M^-l
const maxLevelCap = 64

func (h *Index) Save(dir string) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("hnsw: save: %w", err)
	}

	err := writeFile(filepath.Join(dir, vectorsFile), func(w *binWriter) {
		for _, x := range h.vecs {
			w.f32(x)
		}
	})
	if err != nil {
		return err
	}

	err = writeFile(filepath.Join(dir, idsFile), func(w *binWriter) {
		for _, id := range h.ids {
			w.u64(id)
		}
	})
	if err != nil {
		return err
	}

	return writeFile(filepath.Join(dir, graphFile), func(w *binWriter) {
		// header: same order as Load reads it
		w.u32(uint32(h.dim))
		w.u32(uint32(h.m))
		w.u32(uint32(h.efConstruction))
		w.u32(uint32(len(h.nodes)))
		w.u32(uint32(h.entry)) // -1 (empty index) is written as 0xffffffff
		w.u32(uint32(h.maxLevel))

		for _, nd := range h.nodes {
			w.u32(uint32(nd.level))
			for _, links := range nd.neighbors {
				w.u32(uint32(len(links)))
				for _, nb := range links {
					w.u32(nb)
				}
			}
		}
	})
}

func Load(dir string) (*Index, error) {
	f, err := os.Open(filepath.Join(dir, graphFile))
	if err != nil {
		return nil, fmt.Errorf("hnsw: load: %w", err)
	}
	defer f.Close()
	r := &binReader{r: bufio.NewReader(f)}

	// header: same order as Save writes it
	dim := int(r.u32())
	m := int(r.u32())
	efc := int(r.u32())
	n := int(r.u32())
	entry := int32(r.u32())
	maxLevel := int(r.u32())
	if r.err != nil {
		return nil, corrupt("graph header: %v", r.err)
	}
	emptyOK := n == 0 && entry == -1
	entryOK := n > 0 && entry >= 0 && int(entry) < n
	if dim <= 0 || m < 2 || maxLevel > maxLevelCap || !(emptyOK || entryOK) {
		return nil, corrupt("bad graph header: dim=%d m=%d n=%d entry=%d maxLevel=%d", dim, m, n, entry, maxLevel)
	}

	// the flat files first: their sizes check n before we allocate n nodes
	vecs, unmap, err := mapVectors(filepath.Join(dir, vectorsFile), n*dim)
	if err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			unmap()
		}
	}()
	idBytes, err := readExact(filepath.Join(dir, idsFile), n*8)
	if err != nil {
		return nil, err
	}

	nodes := make([]node, n)
	for i := range nodes {
		if nodes[i], err = readNode(r, n, maxLevel); err != nil {
			return nil, corrupt("node %d: %v", i, err)
		}
	}
	if r.err != nil {
		return nil, corrupt("graph: %v", r.err)
	}

	h := New(dim, Config{M: m, EfConstruction: efc})
	h.vecs, h.unmap = vecs, unmap
	h.ids = make([]uint64, n)
	for i := range h.ids {
		h.ids[i] = binary.LittleEndian.Uint64(idBytes[i*8:])
	}
	h.nodes = nodes
	h.entry = entry
	h.maxLevel = maxLevel
	ok = true
	return h, nil
}

// readNode reads one node's level and neighbour lists, checking them against n and maxLevel.
// Read errors are left in r.err for the caller.
func readNode(r *binReader, n, maxLevel int) (node, error) {
	level := int(r.u32())
	if level > maxLevel {
		return node{}, fmt.Errorf("level %d > maxLevel %d", level, maxLevel)
	}
	nb := make([][]uint32, level+1)
	for l := range nb {
		count := int(r.u32())
		if count > n {
			return node{}, fmt.Errorf("layer %d: %d links", l, count)
		}
		links := make([]uint32, count)
		for j := range links {
			links[j] = r.u32()
			if int(links[j]) >= n {
				return node{}, fmt.Errorf("layer %d: link to %d, only %d nodes", l, links[j], n)
			}
		}
		nb[l] = links
	}
	return node{level: level, neighbors: nb}, nil
}

func corrupt(format string, args ...any) error {
	return fmt.Errorf("hnsw: load: %w: %s", errs.ErrCorrupt, fmt.Sprintf(format, args...))
}

func readExact(path string, size int) ([]byte, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("hnsw: load: %w", err)
	}
	if len(b) != size {
		return nil, corrupt("%s is %d bytes, want %d", filepath.Base(path), len(b), size)
	}
	return b, nil
}

// writeFile creates path, lets fill write the contents, then flushes and fsyncs.
func writeFile(path string, fill func(*binWriter)) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("hnsw: save: %w", err)
	}
	w := &binWriter{w: bufio.NewWriter(f)}
	fill(w)
	if err := w.w.Flush(); err != nil { // any write error from fill shows up here
		f.Close()
		return fmt.Errorf("hnsw: save %s: %w", filepath.Base(path), err)
	}
	if err := f.Sync(); err != nil { // OS cache -> disk, so a crash right after Save keeps the files
		f.Close()
		return fmt.Errorf("hnsw: save %s: %w", filepath.Base(path), err)
	}
	return f.Close()
}

// binWriter writes little-endian numbers. Its methods don't return errors:
// once a write fails, bufio.Writer remembers the error, ignores later writes,
// and returns it from Flush, which writeFile checks.
type binWriter struct {
	w   *bufio.Writer
	buf [8]byte // scratch space, reused for every number
}

func (b *binWriter) u32(v uint32) {
	binary.LittleEndian.PutUint32(b.buf[:4], v)
	b.w.Write(b.buf[:4])
}

func (b *binWriter) u64(v uint64) {
	binary.LittleEndian.PutUint64(b.buf[:8], v)
	b.w.Write(b.buf[:8])
}

func (b *binWriter) f32(v float32) {
	b.u32(math.Float32bits(v))
}

// binReader reads little-endian numbers. bufio.Reader doesn't remember errors,
// so binReader does: after the first failed read, every read returns 0 and the
// caller checks r.err once at the end.
type binReader struct {
	r   *bufio.Reader
	buf [4]byte
	err error
}

func (b *binReader) u32() uint32 {
	if b.err != nil {
		return 0
	}
	// ReadFull: a plain Read may return fewer than 4 bytes
	if _, b.err = io.ReadFull(b.r, b.buf[:]); b.err != nil {
		return 0
	}
	return binary.LittleEndian.Uint32(b.buf[:])
}
