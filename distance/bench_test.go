package distance

import (
	"fmt"
	"math/rand"
	"testing"
)

// so the compiler can't drop the call
var sink float32

// 128 = SIFT, 768 = a typical text-embedding model.
var dims = []int{128, 768}

func randVec(r *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = r.Float32()
	}
	return v
}

func benchDist(b *testing.B, fn func(a, b []float32) (float32, error)) {
	r := rand.New(rand.NewSource(1))
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
