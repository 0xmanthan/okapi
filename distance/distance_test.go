package distance

import (
	"errors"
	"math"
	"testing"

	"okapi/internal/errs"
)

const eps = 1e-6

func approxEqual(a, b float32) bool {
	return math.Abs(float64(a-b)) < eps
}

func TestL2Squared(t *testing.T) {
	cases := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"hand example", []float32{1, 2, 3}, []float32{4, 6, 3}, 25}, // 9 + 16 + 0
		{"identical", []float32{1, 2, 3}, []float32{1, 2, 3}, 0},
		{"negatives", []float32{-1, -2}, []float32{1, 2}, 20}, // 4 + 16
		{"empty", []float32{}, []float32{}, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := L2Squared(c.a, c.b)
			if err != nil {
				t.Fatal(err)
			}
			if !approxEqual(got, c.want) {
				t.Errorf("got %v, want %v", got, c.want)
			}
		})
	}
}

func TestDot(t *testing.T) {
	cases := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"hand example", []float32{1, 2, 3}, []float32{4, 6, 3}, 25}, // 4 + 12 + 9
		{"orthogonal", []float32{1, 0}, []float32{0, 1}, 0},
		{"opposite", []float32{1, 2}, []float32{-1, -2}, -5},
		{"empty", []float32{}, []float32{}, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := Dot(c.a, c.b)
			if err != nil {
				t.Fatal(err)
			}
			if !approxEqual(got, c.want) {
				t.Errorf("got %v, want %v", got, c.want)
			}
		})
	}
}

func TestLengthMismatch(t *testing.T) {
	a, b := []float32{1, 2, 3}, []float32{1, 2}

	if _, err := L2Squared(a, b); !errors.Is(err, errs.ErrDimMismatch) {
		t.Errorf("L2Squared: got %v, want ErrDimMismatch", err)
	}
	if _, err := Dot(a, b); !errors.Is(err, errs.ErrDimMismatch) {
		t.Errorf("Dot: got %v, want ErrDimMismatch", err)
	}
}

func TestNormalize(t *testing.T) {
	v := []float32{3, 4}
	Normalize(v)

	want := []float32{0.6, 0.8}
	for i := range want {
		if !approxEqual(v[i], want[i]) {
			t.Errorf("v[%d] = %v, want %v", i, v[i], want[i])
		}
	}

	self, _ := Dot(v, v)
	if !approxEqual(self, 1) {
		t.Errorf("Dot(v, v) = %v after Normalize, want 1", self)
	}
}

func TestNormalizeZero(t *testing.T) {
	v := []float32{0, 0, 0}
	Normalize(v)
	for i, x := range v {
		if x != 0 || math.IsNaN(float64(x)) {
			t.Errorf("v[%d] = %v, want 0 (no divide by zero)", i, x)
		}
	}
}

func TestCosineViaDot(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{2, 4, 6}
	Normalize(a)
	Normalize(b)
	got, _ := Dot(a, b)
	if !approxEqual(got, 1) {
		t.Errorf("cosine of parallel vectors = %v, want 1", got)
	}
}
