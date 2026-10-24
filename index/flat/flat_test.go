package flat

import (
	"errors"
	"math/rand"
	"sort"
	"testing"

	"okapi/internal/errs"
)

// 2-D points you can check by eye. Squared distances from q=[0,0]:
// id0 → 0, id1 → 1, id3 → 4, id2 → 50.
func smallIndex(t *testing.T) *Flat {
	t.Helper()
	f := New(2)
	pts := map[uint64][]float32{0: {0, 0}, 1: {1, 0}, 2: {5, 5}, 3: {0, 2}}
	for id := uint64(0); id < 4; id++ {
		if err := f.Add(id, pts[id]); err != nil {
			t.Fatal(err)
		}
	}
	return f
}

func TestSearchTopK(t *testing.T) {
	f := smallIndex(t)
	got, err := f.Search([]float32{0, 0}, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := []Result{{ID: 0, Dist: 0}, {ID: 1, Dist: 1}, {ID: 3, Dist: 4}}
	if len(got) != len(want) {
		t.Fatalf("got %d results %v, want %v", len(got), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("result %d = %+v, want %+v (results must be nearest first)", i, got[i], want[i])
		}
	}
}

func TestSearchKLargerThanIndex(t *testing.T) {
	f := smallIndex(t)
	got, err := f.Search([]float32{0, 0}, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 4 {
		t.Fatalf("got %d results, want 4 (only 4 vectors stored)", len(got))
	}
}

func TestSearchEmptyIndex(t *testing.T) {
	got, err := New(2).Search([]float32{0, 0}, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("got %v from an empty index, want no results", got)
	}
}

func TestAddWrongDim(t *testing.T) {
	err := New(2).Add(0, []float32{1, 2, 3})
	if !errors.Is(err, errs.ErrDimMismatch) {
		t.Fatalf("got %v, want errs.ErrDimMismatch", err)
	}
}

func TestSearchWrongDim(t *testing.T) {
	_, err := smallIndex(t).Search([]float32{1, 2, 3}, 2)
	if !errors.Is(err, errs.ErrDimMismatch) {
		t.Fatalf("got %v, want errs.ErrDimMismatch", err)
	}
}

func TestSearchBadK(t *testing.T) {
	for _, k := range []int{0, -1} {
		_, err := smallIndex(t).Search([]float32{0, 0}, k)
		if !errors.Is(err, errs.ErrInvalidArgument) {
			t.Errorf("k=%d: got %v, want errs.ErrInvalidArgument", k, err)
		}
	}
}

// check against just sorting every distance
func TestSearchMatchesSort(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	const n, dim, k = 500, 16, 10
	f := New(dim)
	vecs := make([][]float32, n)
	for i := range vecs {
		vecs[i] = make([]float32, dim)
		for j := range vecs[i] {
			vecs[i][j] = r.Float32()
		}
		if err := f.Add(uint64(i), vecs[i]); err != nil {
			t.Fatal(err)
		}
	}

	for trial := 0; trial < 20; trial++ {
		q := make([]float32, dim)
		for j := range q {
			q[j] = r.Float32()
		}

		all := make([]Result, n)
		for i, v := range vecs {
			var d float32
			for j := range v {
				diff := q[j] - v[j]
				d += diff * diff
			}
			all[i] = Result{ID: uint64(i), Dist: d}
		}
		sort.Slice(all, func(a, b int) bool { return all[a].Dist < all[b].Dist })

		got, err := f.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < k; i++ {
			if got[i].ID != all[i].ID {
				t.Fatalf("trial %d, rank %d: got id %d, want id %d", trial, i, got[i].ID, all[i].ID)
			}
		}
	}
}
