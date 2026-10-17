package distance

import (
	"fmt"
	"okapi/internal/errs"
)

func L2Squared(a, b []float32) (float32, error) {

	n := len(a)

	if n != len(b) {
		return 0, fmt.Errorf("distance: %w: %d vs %d", errs.ErrDimMismatch, len(a), len(b))
	}

	i := 0

	s0, s1, s2, s3 := float32(0), float32(0), float32(0), float32(0)

	for ; i+4 <= n; i += 4 {
		s0 += (a[i] - b[i]) * (a[i] - b[i])
		s1 += (a[i+1] - b[i+1]) * (a[i+1] - b[i+1])
		s2 += (a[i+2] - b[i+2]) * (a[i+2] - b[i+2])
		s3 += (a[i+3] - b[i+3]) * (a[i+3] - b[i+3])
	}

	for ; i < n; i++ {
		s0 += (a[i] - b[i]) * (a[i] - b[i])
	}

	return s0 + s1 + s2 + s3, nil
}

func Dot(a, b []float32) (float32, error) {
	n := len(a)

	if n != len(b) {
		return 0, fmt.Errorf("distance: %w: %d vs %d", errs.ErrDimMismatch, len(a), len(b))
	}

	i := 0

	s0, s1, s2, s3 := float32(0), float32(0), float32(0), float32(0)

	for ; i+4 <= n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}

	for ; i < n; i++ {
		s0 += a[i] * b[i]
	}

	return s0 + s1 + s2 + s3, nil
}
