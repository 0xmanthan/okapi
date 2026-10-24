package hnsw

import (
	"fmt"
	"os"
	"slices"
	"sync"
	"testing"

	"okapi/dataset"
)

const siftDir = "../../data/siftsmall/"

var sinkResults []Result

type siftData struct {
	base, query [][]float32
	gt          [][]int32
	idx         *Index
}

var (
	siftOnce   sync.Once
	sift       siftData
	siftLoadEr error
)

// loadSift reads siftsmall and builds one index, shared by every benchmark in the run.
func loadSift(b *testing.B) siftData {
	b.Helper()
	if _, err := os.Stat(siftDir); err != nil {
		b.Skipf("siftsmall not found at %s; download it first", siftDir)
	}
	siftOnce.Do(func() {
		if sift.base, siftLoadEr = dataset.ReadFVecs(siftDir + "siftsmall_base.fvecs"); siftLoadEr != nil {
			return
		}
		if sift.query, siftLoadEr = dataset.ReadFVecs(siftDir + "siftsmall_query.fvecs"); siftLoadEr != nil {
			return
		}
		if sift.gt, siftLoadEr = dataset.ReadIVecs(siftDir + "siftsmall_groundtruth.ivecs"); siftLoadEr != nil {
			return
		}
		sift.idx, siftLoadEr = buildIndex(sift.base)
	})
	if siftLoadEr != nil {
		b.Fatal(siftLoadEr)
	}
	return sift
}

func buildIndex(base [][]float32) (*Index, error) {
	h := New(len(base[0]), Config{M: 16, EfConstruction: 200, Seed: 1})
	for i, v := range base {
		if err := h.Add(uint64(i), v); err != nil {
			return nil, err
		}
	}
	return h, nil
}

// BenchmarkBuild is one full siftsmall build (10k × 128-d, M=16, efConstruction=200) per op.
func BenchmarkBuild(b *testing.B) {
	data := loadSift(b)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := buildIndex(data.base); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(len(data.base))*float64(b.N)/b.Elapsed().Seconds(), "inserts/s")
}

// BenchmarkSearch is one k=10 query per op, cycling through the 100 siftsmall queries.
// recall@10 is reported next to ns/op so speed and accuracy are read together.
func BenchmarkSearch(b *testing.B) {
	data := loadSift(b)
	const k = 10
	for _, ef := range []int{10, 20, 50, 100, 200} {
		b.Run(fmt.Sprintf("ef=%d", ef), func(b *testing.B) {
			b.ReportAllocs()
			hits, n := 0, 0
			for b.Loop() {
				q := n % len(data.query)
				res, err := data.idx.Search(data.query[q], k, ef)
				if err != nil {
					b.Fatal(err)
				}
				hits += countHits(res, data.gt[q][:k])
				sinkResults = res
				n++
			}
			b.ReportMetric(float64(hits)/float64(n*k), "recall@10")
			b.ReportMetric(float64(n)/b.Elapsed().Seconds(), "QPS")
		})
	}
}

func countHits(res []Result, truth []int32) int {
	hits := 0
	for _, r := range res {
		if slices.Contains(truth, int32(r.ID)) {
			hits++
		}
	}
	return hits
}
