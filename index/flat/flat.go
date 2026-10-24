package flat

import (
	"container/heap"
	"fmt"
	"okapi/distance"
	"okapi/internal/errs"
)

type Result struct {
	ID   uint64
	Dist float32
}

type Flat struct {
	dim  int
	ids  []uint64
	vecs [][]float32
}

func New(dim int) *Flat {
	return &Flat{
		dim: dim,
	}
}

func (f *Flat) Add(id uint64, vec []float32) error {
	if len(vec) != f.dim {
		return fmt.Errorf("flat: %w: got %d, want %d", errs.ErrDimMismatch, len(vec), f.dim)
	}

	// caller may reuse vec
	v := make([]float32, len(vec))
	copy(v, vec)

	f.ids = append(f.ids, id)
	f.vecs = append(f.vecs, v)
	return nil
}

// nearest first
func (f *Flat) Search(q []float32, k int) ([]Result, error) {
	if k <= 0 {
		return nil, fmt.Errorf("flat: %w: k=%d", errs.ErrInvalidArgument, k)
	}
	if len(q) != f.dim {
		return nil, fmt.Errorf("flat: %w: got %d, want %d", errs.ErrDimMismatch, len(q), f.dim)
	}

	// k nearest so far, (*h)[0] is the farthest
	h := &maxHeap{}

	for i, v := range f.vecs {
		d, _ := distance.L2Squared(q, v)

		if h.Len() < k {
			heap.Push(h, Result{ID: f.ids[i], Dist: d})
		} else if d < (*h)[0].Dist {
			(*h)[0] = Result{ID: f.ids[i], Dist: d}
			heap.Fix(h, 0)
		}
	}

	out := make([]Result, h.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(h).(Result)
	}
	return out, nil
}
