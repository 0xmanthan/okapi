package segment

import (
	"bytes"
	"errors"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"okapi/index/hnsw"
	"okapi/internal/errs"
	"okapi/internal/testutil"
	"okapi/storage/wal"
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
	walPath := filepath.Join(testutil.OutDir(t), "wal-000001.log")
	cfg := hnsw.Config{M: 16, EfConstruction: 100, Seed: 1}
	rng := rand.New(rand.NewPCG(1, 2))

	s, err := Open(walPath, dim, cfg)
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

	s, err = Open(walPath, dim, cfg)
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
	s, err = Open(walPath, dim, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if got := s.idx.Len(); got != n+1 {
		t.Fatalf("after second reopen: %d nodes, want %d", got, n+1)
	}
}

func fileSize(t *testing.T, path string) int64 {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return info.Size()
}

func TestSealSameResults(t *testing.T) {
	const n, dim = 1000, 16
	dir := testutil.OutDir(t)
	wal1 := filepath.Join(dir, "wal-000001.log")
	wal2 := filepath.Join(dir, "wal-000002.log")
	segDir := filepath.Join(dir, "seg-000001")
	cfg := hnsw.Config{M: 16, EfConstruction: 100, Seed: 1}
	rng := rand.New(rand.NewPCG(1, 2))

	s, err := Open(wal1, dim, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
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

	old, err := s.Seal(segDir, wal2)
	if err != nil {
		t.Fatal(err)
	}
	if old != wal1 {
		t.Errorf("Seal returned %q, want %q", old, wal1)
	}

	sealed, err := hnsw.Load(segDir)
	if err != nil {
		t.Fatal(err)
	}
	for i, q := range queries {
		got, err := sealed.Search(q, 10, 50)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, before[i]) {
			t.Errorf("query %d: sealed %v, before seal %v", i, got, before[i])
		}
	}

	// the mutable side starts over: empty index, empty WAL
	if s.Len() != 0 {
		t.Errorf("after seal: mutable has %d nodes, want 0", s.Len())
	}
	if got := fileSize(t, wal2); got != 0 {
		t.Errorf("new WAL is %d bytes, want 0", got)
	}

	// new writes go to the new WAL only; the old one is frozen
	oldSize := fileSize(t, wal1)
	if err := s.Upsert(n, randomVec(rng, dim)); err != nil {
		t.Fatal(err)
	}
	if fileSize(t, wal1) != oldSize {
		t.Error("upsert after seal changed the old WAL")
	}
	if fileSize(t, wal2) == 0 {
		t.Error("upsert after seal didn't reach the new WAL")
	}
}

// sealing onto a WAL that already exists would mix two segments' records in one file
func TestSealRefusesExistingWAL(t *testing.T) {
	dir := testutil.OutDir(t)
	wal1 := filepath.Join(dir, "wal-000001.log")
	wal2 := filepath.Join(dir, "wal-000002.log")
	cfg := hnsw.Config{M: 16, EfConstruction: 100, Seed: 1}

	s, err := Open(wal1, 2, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Upsert(1, []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(wal2, []byte("someone else's records"), 0644); err != nil {
		t.Fatal(err)
	}

	if _, err := s.Seal(filepath.Join(dir, "seg-000001"), wal2); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("err = %v, want ErrInvalidArgument", err)
	}
	// nothing changed: same data, still writing to the old WAL
	if s.Len() != 1 {
		t.Errorf("after refused seal: %d nodes, want 1", s.Len())
	}
	if err := s.Upsert(2, []float32{3, 4}); err != nil {
		t.Fatal(err)
	}
	if recs, _, err := wal.ReadAll(wal1); err != nil || len(recs) != 2 {
		t.Errorf("old WAL: %d records, err %v; want 2, nil", len(recs), err)
	}
}
