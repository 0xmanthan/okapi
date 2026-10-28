package store

import (
	"math/rand/v2"
	"slices"
	"testing"

	"okapi/index/flat"
	"okapi/index/hnsw"
)

type fixed []hnsw.Result

func (f fixed) Search([]float32, int, int) ([]hnsw.Result, error) { return f, nil }

func TestSearchMerges(t *testing.T) {
	segs := []Searcher{
		fixed{{ID: 1, Dist: 0.5}, {ID: 2, Dist: 3}},
		fixed{{ID: 3, Dist: 0.1}, {ID: 4, Dist: 2}},
		fixed{}, // empty segment
	}
	got, err := Search(segs, nil, 3, 10) // k=3; ef doesn't matter for fixed results
	if err != nil {
		t.Fatal(err)
	}
	want := []hnsw.Result{{ID: 3, Dist: 0.1}, {ID: 1, Dist: 0.5}, {ID: 4, Dist: 2}}
	if !slices.Equal(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func randomVecs(rng *rand.Rand, n, dim int) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		for j := range v {
			v[j] = rng.Float32()
		}
		out[i] = v
	}
	return out
}

// recall@k: of the k true nearest per query, the fraction search found
func recall(t *testing.T, k int, search func([]float32) []hnsw.Result, exact *flat.Flat, queries [][]float32) float64 {
	t.Helper()
	hits := 0
	for _, q := range queries {
		truth, err := exact.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range search(q) {
			if slices.ContainsFunc(truth, func(x flat.Result) bool { return x.ID == r.ID }) {
				hits++
			}
		}
	}
	return float64(hits) / float64(len(queries)*k)
}

// splitting the data into 3 segments shouldn't cost recall
func TestThreeSegmentsVsOne(t *testing.T) {
	const (
		n   = 6000 // vectors stored
		dim = 16
		k   = 10 // results per query
		ef  = 50 // search beam width
	)
	rng := rand.New(rand.NewPCG(1, 2))
	vecs := randomVecs(rng, n, dim)
	queries := randomVecs(rng, 100, dim)
	cfg := hnsw.Config{M: 16, EfConstruction: 100, Seed: 1}

	exact := flat.New(dim)
	one := hnsw.New(dim, cfg)
	three := []*hnsw.Index{hnsw.New(dim, cfg), hnsw.New(dim, cfg), hnsw.New(dim, cfg)}
	for i, v := range vecs {
		for _, err := range []error{exact.Add(uint64(i), v), one.Add(uint64(i), v), three[i%3].Add(uint64(i), v)} {
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	segs := []Searcher{three[0], three[1], three[2]}

	rOne := recall(t, k, func(q []float32) []hnsw.Result {
		r, err := one.Search(q, k, ef)
		if err != nil {
			t.Fatal(err)
		}
		return r
	}, exact, queries)
	rThree := recall(t, k, func(q []float32) []hnsw.Result {
		r, err := Search(segs, q, k, ef)
		if err != nil {
			t.Fatal(err)
		}
		return r
	}, exact, queries)

	t.Logf("recall@%d: one index %.3f, three segments %.3f", k, rOne, rThree)
	if rThree < rOne-0.02 {
		t.Errorf("three segments recall %.3f is worse than one index %.3f", rThree, rOne)
	}
}
