// recall@k and QPS on siftsmall. run from repo root:
//
//	go run ./cmd/eval -show 0
//	go run ./cmd/eval -index hnsw -ef 10,20,50,100,200
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"

	"okapi/dataset"
	"okapi/index/flat"
	"okapi/index/hnsw"
)

type result struct {
	ID   uint64
	Dist float32
}

const dataDir = "data/siftsmall/"

func main() {
	k := flag.Int("k", 10, "neighbours to return per query")
	show := flag.Int("show", 2, "print the full result table for this many queries")
	indexType := flag.String("index", "flat", "flat or hnsw")
	efList := flag.String("ef", "20", "hnsw: efSearch, or a comma-separated list to sweep")
	m := flag.Int("m", 16, "hnsw: links per node per layer")
	efc := flag.Int("efc", 200, "hnsw: efConstruction")
	flag.Parse()
	efs := parseInts(*efList)

	section("1. Dataset")
	base := mustFVecs(dataDir + "siftsmall_base.fvecs")
	query := mustFVecs(dataDir + "siftsmall_query.fvecs")
	gt, err := dataset.ReadIVecs(dataDir + "siftsmall_groundtruth.ivecs")
	if err != nil {
		log.Fatal(err)
	}
	dim := len(base[0])
	lo, hi := valueRange(base)
	fmt.Printf("base          %d vectors × %d dims\n", len(base), dim)
	fmt.Printf("queries       %d\n", len(query))
	fmt.Printf("ground truth  %d rows × %d ids (we use the first %d)\n", len(gt), len(gt[0]), *k)
	fmt.Printf("value range   %.0f … %.0f\n", lo, hi)
	fmt.Printf("base[0][:8]   %v\n", base[0][:8])
	fmt.Printf("query[0][:8]  %v\n", query[0][:8])

	section("2. Build " + *indexType + " index")
	buildStart := time.Now()
	var search func(q []float32, ef int) ([]result, error)
	switch *indexType {
	case "flat":
		idx := flat.New(dim)
		for i, v := range base {
			if err := idx.Add(uint64(i), v); err != nil {
				log.Fatal(err)
			}
		}
		search = func(q []float32, _ int) ([]result, error) {
			rs, err := idx.Search(q, *k)
			out := make([]result, len(rs))
			for i, r := range rs {
				out[i] = result(r)
			}
			return out, err
		}
	case "hnsw":
		idx := hnsw.New(dim, hnsw.Config{M: *m, EfConstruction: *efc, Seed: 1})
		for i, v := range base {
			if err := idx.Add(uint64(i), v); err != nil {
				log.Fatal(err)
			}
		}
		fmt.Printf("params        M=%d efConstruction=%d efSearch=%v\n", *m, *efc, efs)
		search = func(q []float32, ef int) ([]result, error) {
			rs, err := idx.Search(q, *k, ef)
			out := make([]result, len(rs))
			for i, r := range rs {
				out[i] = result(r)
			}
			return out, err
		}
	default:
		log.Fatalf("unknown -index %q: want flat or hnsw", *indexType)
	}
	fmt.Printf("build time    %v\n", time.Since(buildStart))
	fmt.Printf("vector memory %.2f MB  (%d × %d × 4 bytes)\n",
		float64(len(base)*dim*4)/1e6, len(base), dim)

	if *indexType == "hnsw" && len(efs) > 1 {
		sweep(search, query, gt, *k, efs)
		return
	}

	section("3. Search")
	results, lat, elapsed := runQueries(search, query, efs[0])
	sorted := slices.Clone(lat)
	slices.Sort(sorted)
	fmt.Printf("queries       %d in %v\n", len(query), elapsed)
	fmt.Printf("QPS           %.0f\n", float64(len(query))/elapsed.Seconds())
	fmt.Printf("latency       min %v · p50 %v · p99 %v · max %v\n",
		sorted[0], sorted[len(sorted)/2], sorted[len(sorted)*99/100], sorted[len(sorted)-1])
	if *indexType == "flat" {
		distances := len(base) * len(query)
		fmt.Printf("distances     %d per query, %d total → %.1f ns per distance\n",
			len(base), distances, float64(elapsed.Nanoseconds())/float64(distances))
	}

	section(fmt.Sprintf("4. Recall@%d", *k))
	perQuery, hits := countHits(results, gt, *k)
	fmt.Printf("recall@%d     %.4f  (%d of %d true neighbours found)\n",
		*k, float64(hits)/float64(len(query)**k), hits, len(query)**k)

	fmt.Println("hits per query:")
	for h := *k; h >= 0; h-- {
		n := 0
		for _, x := range perQuery {
			if x == h {
				n++
			}
		}
		if n > 0 {
			fmt.Printf("  %2d/%d  %3d queries  %s\n", h, *k, n, bar(n, len(query)))
		}
	}
	worst := 0
	for q := range perQuery {
		if perQuery[q] < perQuery[worst] {
			worst = q
		}
	}
	fmt.Printf("worst query   #%d with %d/%d\n", worst, perQuery[worst], *k)

	for q := 0; q < *show && q < len(query); q++ {
		section(fmt.Sprintf("5. Query #%d in detail", q))
		truth := idSet(gt[q][:*k])
		fmt.Printf("%4s  %6s  %10s  %8s  %s\n", "rank", "our id", "distance²", "true id", "in true top-k?")
		for i, r := range results[q] {
			mark := "✓"
			if !truth[int32(r.ID)] {
				mark = "✗"
			}
			fmt.Printf("%4d  %6d  %10.0f  %8d  %s\n", i+1, r.ID, r.Dist, gt[q][i], mark)
		}
		fmt.Printf("nearest is %.0f away, %d-th is %.0f away, searched in %v\n",
			results[q][0].Dist, *k, results[q][len(results[q])-1].Dist, lat[q])
	}
}

// best of 5, one pass is a few ms so a single run is noisy
const sweepRuns = 5

func sweep(search func([]float32, int) ([]result, error), query [][]float32, gt [][]int32, k int, efs []int) {
	section("3. efSearch sweep")
	fmt.Printf("%5s  %9s  %8s  %9s  %9s  %s\n", "ef", fmt.Sprintf("recall@%d", k), "QPS", "p50", "p99", "worst query")
	for _, ef := range efs {
		var results [][]result
		var lat []time.Duration
		best := time.Duration(math.MaxInt64)
		for range sweepRuns {
			r, l, elapsed := runQueries(search, query, ef)
			if elapsed < best {
				results, lat, best = r, l, elapsed
			}
		}
		perQuery, hits := countHits(results, gt, k)
		slices.Sort(lat)
		fmt.Printf("%5d  %9.4f  %8.0f  %9v  %9v  %d/%d\n",
			ef, float64(hits)/float64(len(query)*k), float64(len(query))/best.Seconds(),
			lat[len(lat)/2], lat[len(lat)*99/100], slices.Min(perQuery), k)
	}
}

// timed per query for p50/p99
func runQueries(search func([]float32, int) ([]result, error), query [][]float32, ef int) ([][]result, []time.Duration, time.Duration) {
	results := make([][]result, len(query))
	lat := make([]time.Duration, len(query))
	total := time.Now()
	for q := range query {
		t := time.Now()
		var err error
		results[q], err = search(query[q], ef)
		lat[q] = time.Since(t)
		if err != nil {
			log.Fatal(err)
		}
	}
	return results, lat, time.Since(total)
}

func countHits(results [][]result, gt [][]int32, k int) ([]int, int) {
	perQuery := make([]int, len(results))
	hits := 0
	for q, rs := range results {
		truth := idSet(gt[q][:k])
		for _, r := range rs {
			if truth[int32(r.ID)] {
				perQuery[q]++
			}
		}
		hits += perQuery[q]
	}
	return perQuery, hits
}

func parseInts(s string) []int {
	var out []int
	for _, f := range strings.Split(s, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil || n <= 0 {
			log.Fatalf("bad -ef value %q: want positive integers like 10,20,50", f)
		}
		out = append(out, n)
	}
	return out
}

func section(title string) {
	fmt.Printf("\n── %s ──\n", title)
}

func mustFVecs(path string) [][]float32 {
	v, err := dataset.ReadFVecs(path)
	if err != nil {
		log.Fatal(err)
	}
	return v
}

func valueRange(vecs [][]float32) (lo, hi float32) {
	lo, hi = vecs[0][0], vecs[0][0]
	for _, v := range vecs {
		for _, x := range v {
			lo, hi = min(lo, x), max(hi, x)
		}
	}
	return lo, hi
}

func idSet(ids []int32) map[int32]bool {
	s := make(map[int32]bool, len(ids))
	for _, id := range ids {
		s[id] = true
	}
	return s
}

func bar(n, of int) string {
	w := n * 40 / of
	if w == 0 && n > 0 {
		w = 1
	}
	b := make([]rune, w)
	for i := range b {
		b[i] = '█'
	}
	return string(b)
}
