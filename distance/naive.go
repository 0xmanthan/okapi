package distance

import (
	"fmt"
	"math"
	"okapi/internal/errs"
)

// l2Naive and dotNaive are the reference versions the unrolled ones are tested against.
func l2Naive(a, b []float32) (float32, error) {
	sum := float32(0)

	n := len(a)
	if n != len(b) {
		return 0, fmt.Errorf("distance: %w: %d vs %d", errs.ErrDimMismatch, len(a), len(b))
	}

	for i := range n {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return sum, nil
}

func dotNaive(a, b []float32) (float32, error) {
	dot := float32(0)
	n := len(a)
	if n != len(b) {
		return 0, fmt.Errorf("distance: %w: %d vs %d", errs.ErrDimMismatch, len(a), len(b))
	}

	for i := range n {
		dt := a[i] * b[i]
		dot += dt
	}

	return dot, nil
}

// Normalize scales v to unit length in place, so Dot then gives cosine similarity.
// A zero vector is left unchanged.
func Normalize(v []float32) {
	sumSq := float32(0)

	for _, x := range v {
		sumSq += (x * x)
	}

	norm := float32(math.Sqrt(float64(sumSq)))

	if norm == 0 {
		return
	}

	for i := range v {
		v[i] /= norm
	}
}
