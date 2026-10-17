package distance

import (
	"fmt"
	"math/rand"
	"testing"
)

// sink stops the compiler from deleting a call whose result is never used.
var sink float32

// dims: 128 = SIFT, 768 = a typical text-embedding model.
var dims = []int{128, 768}

func randVec(r *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = r.Float32()
	}
	return v
}

// benchDist runs fn on two fixed random vectors of each size in dims.
func benchDist(b *testing.B, fn func(a, b []float32) (float32, error)) {
	r := rand.New(rand.NewSource(1)) // fixed seed: same vectors every run
	for _, d := range dims {
		x, y := randVec(r, d), randVec(r, d)
		b.Run(fmt.Sprintf("d=%d", d), func(b *testing.B) {
			for b.Loop() {
				s, _ := fn(x, y)
				sink = s
			}
		})
	}
}

func BenchmarkL2Naive(b *testing.B)     { benchDist(b, l2Naive) }
func BenchmarkL2Unrolled(b *testing.B)  { benchDist(b, L2Squared) }
func BenchmarkDotNaive(b *testing.B)    { benchDist(b, dotNaive) }
func BenchmarkDotUnrolled(b *testing.B) { benchDist(b, Dot) }
