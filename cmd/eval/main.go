// Command eval runs the siftsmall queries against an index and reports recall@k and QPS.
// Run from the repo root:
//
//	go run ./cmd/eval            // summary + details for the first 2 queries
//	go run ./cmd/eval -show 5    // details for the first 5 queries
//	go run ./cmd/eval -show 0    // summary only
package main

import (
	"flag"
	"fmt"
	"log"
	"slices"
	"time"

	"okapi/dataset"
	"okapi/index/flat"
)

const dataDir = "data/siftsmall/"

func main() {
	k := flag.Int("k", 10, "neighbours to return per query")
	show := flag.Int("show", 2, "print the full result table for this many queries")
	flag.Parse()

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

	section("2. Build flat index")
	buildStart := time.Now()
	idx := flat.New(dim)
	for i, v := range base {
		if err := idx.Add(uint64(i), v); err != nil {
			log.Fatal(err)
		}
	}
	fmt.Printf("build time    %v\n", time.Since(buildStart))
	fmt.Printf("vector memory %.2f MB  (%d × %d × 4 bytes)\n",
		float64(len(base)*dim*4)/1e6, len(base), dim)

	// Time each query separately to get the latency spread.
	section("3. Search")
	results := make([][]flat.Result, len(query))
	lat := make([]time.Duration, len(query))
	total := time.Now()
	for q := range query {
		t := time.Now()
		results[q], err = idx.Search(query[q], *k)
		lat[q] = time.Since(t)
		if err != nil {
			log.Fatal(err)
		}
	}
	elapsed := time.Since(total)

	sorted := slices.Clone(lat)
	slices.Sort(sorted)
	distances := len(base) * len(query)
	fmt.Printf("queries       %d in %v\n", len(query), elapsed)
	fmt.Printf("QPS           %.0f\n", float64(len(query))/elapsed.Seconds())
	fmt.Printf("latency       min %v · p50 %v · p99 %v · max %v\n",
		sorted[0], sorted[len(sorted)/2], sorted[len(sorted)*99/100], sorted[len(sorted)-1])
	fmt.Printf("distances     %d per query, %d total → %.1f ns per distance\n",
		len(base), distances, float64(elapsed.Nanoseconds())/float64(distances))

	section(fmt.Sprintf("4. Recall@%d", *k))
	perQuery := make([]int, len(query))
	hits := 0
	for q := range query {
		truth := idSet(gt[q][:*k])
		for _, r := range results[q] {
			if truth[int32(r.ID)] {
				perQuery[q]++
			}
		}
		hits += perQuery[q]
	}
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
