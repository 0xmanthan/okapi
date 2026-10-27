package segment

import (
	"bytes"
	"math/rand/v2"
	"okapi/index/hnsw"
	"slices"
	"testing"
)

func TestUpsertRoundTrip(t *testing.T) {
	id := uint64(7)
	vec := []float32{1.0, 0.5}

	b := encodeUpsert(id, vec)
	want := []byte{
		0x07, 0, 0, 0, 0, 0, 0, 0,
		0x00, 0x00, 0x80, 0x3f, // 1.0
		0x00, 0x00, 0x00, 0x3f, // 0.5
	}
	if !bytes.Equal(b, want) {
		t.Fatalf("encoded % x, want % x", b, want)
	}

	gotID, gotVec, err := decodeUpsert(b)
	if err != nil {
		t.Fatal(err)
	}
	if gotID != id || !slices.Equal(gotVec, vec) {
		t.Fatalf("decoded %d %v, want %d %v", gotID, gotVec, id, vec)
	}
}

func TestDecodeUpsertBadLength(t *testing.T) {
	for _, n := range []int{0, 7, 9, 13} {
		if _, _, err := decodeUpsert(make([]byte, n)); err == nil {
			t.Errorf("len %d: no error", n)
		}
	}
}

func randomVec(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rng.Float32()
	}
	return v
}

func searchAll(t *testing.T, s *Segment, queries [][]float32) [][]hnsw.Result {
	t.Helper()
	out := make([][]hnsw.Result, len(queries))
	for i, q := range queries {
		res, err := s.Search(q, 10, 50)
		if err != nil {
			t.Fatal(err)
		}
		out[i] = res
	}
	return out
}

// same inserts + same seed = same graph, so results must match exactly
func TestReopenSameResults(t *testing.T) {
	const n, dim = 1000, 16
	dir := t.TempDir()
	cfg := hnsw.Config{M: 16, EfConstruction: 100, Seed: 1}
	rng := rand.New(rand.NewPCG(1, 2))

	s, err := Open(dir, dim, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := range n {
		if err := s.Upsert(uint64(i), randomVec(rng, dim)); err != nil {
			t.Fatal(err)
		}
	}
	queries := make([][]float32, 10)
	for i := range queries {
		queries[i] = randomVec(rng, dim)
	}
	before := searchAll(t, s, queries)
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}

	s, err = Open(dir, dim, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if got := s.idx.Len(); got != n {
		t.Fatalf("after reopen: %d nodes, want %d", got, n)
	}
	after := searchAll(t, s, queries)
	for i := range queries {
		if !slices.Equal(before[i], after[i]) {
			t.Errorf("query %d: before %v, after %v", i, before[i], after[i])
		}
	}

	// appends after recovery land cleanly too
	if err := s.Upsert(n, randomVec(rng, dim)); err != nil {
		t.Fatal(err)
	}
	s.Close()
	s, err = Open(dir, dim, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if got := s.idx.Len(); got != n+1 {
		t.Fatalf("after second reopen: %d nodes, want %d", got, n+1)
	}
}
