package hnsw

import (
	"cmp"
	"errors"
	"math"
	"math/rand"
	"slices"
	"testing"

	"okapi/index/flat"
	"okapi/internal/errs"
)

func TestVecIsSlabView(t *testing.T) {
	h := New(3, Config{
		M:              2,
		EfConstruction: 2,
		Seed:           10,
	})
	h.vecs = []float32{1, 2, 3, 4, 5, 6}

	got := h.vec(1)
	if !slices.Equal(got, []float32{4, 5, 6}) {
		t.Fatalf("vec(1) = %v, want [4 5 6]", got)
	}

	got[0] = 99
	if h.vecs[3] != 99 {
		t.Errorf("after vec(1)[0] = 99, h.vecs[3] = %v; vec must return a view, not a copy", h.vecs[3])
	}
}

func TestRandomLevelDistribution(t *testing.T) {
	const m, draws = 16, 200_000
	h := New(4, Config{M: m, EfConstruction: 10, Seed: 1})

	atLeast := make([]int, 3)
	for range draws {
		level := h.randomLevel()
		if level < 0 {
			t.Fatalf("randomLevel() = %d, want >= 0", level)
		}
		for l := range min(level+1, len(atLeast)) {
			atLeast[l]++
		}
	}

	// P(level >= l) = M^-l. Tolerance is about 5 standard deviations of the count.
	cases := []struct {
		level int
		want  float64
		tol   float64
	}{
		{1, 1.0 / m, 0.05},
		{2, 1.0 / (m * m), 0.20},
	}
	for _, c := range cases {
		got := float64(atLeast[c.level]) / draws
		t.Logf("P(level>=%d) = %.5f, want %.5f", c.level, got, c.want)
		if math.Abs(got-c.want)/c.want > c.tol {
			t.Errorf("P(level>=%d) = %.5f, want %.5f ± %.0f%%", c.level, got, c.want, c.tol*100)
		}
	}
}

// 1-D, layer 0 only. edges are undirected node pairs.
func lineGraph(pts []float32, edges [][2]uint32) *Index {
	h := New(1, Config{M: 4, EfConstruction: 10})
	h.vecs = slices.Clone(pts)
	h.ids = make([]uint64, len(pts))
	h.nodes = make([]node, len(pts))
	for i := range pts {
		h.ids[i] = uint64(i)
		h.nodes[i] = node{level: 0, neighbors: [][]uint32{nil}}
	}
	for _, e := range edges {
		a, b := e[0], e[1]
		h.nodes[a].neighbors[0] = append(h.nodes[a].neighbors[0], b)
		h.nodes[b].neighbors[0] = append(h.nodes[b].neighbors[0], a)
	}
	h.entry = 0
	return h
}

func nodeIDs(cands []candidate) []uint32 {
	out := make([]uint32, len(cands))
	for i, c := range cands {
		out[i] = c.id
	}
	return out
}

// Squared distances to q=8: 49 36 16 4 1 16 36.
func TestSearchLayerLine(t *testing.T) {
	h := lineGraph(
		[]float32{1, 2, 4, 6, 9, 12, 14},
		[][2]uint32{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}},
	)
	q := []float32{8}

	cases := []struct {
		ef   int
		want []uint32
	}{
		{2, []uint32{4, 3}},
		{1, []uint32{4}},
	}
	for _, c := range cases {
		got := nodeIDs(h.searchLayer(q, []uint32{5}, c.ef, 0))
		if !slices.Equal(got, c.want) {
			t.Errorf("ef=%d: got nodes %v, want %v", c.ef, got, c.want)
		}
	}
}

// Every neighbour of B is farther from q than B, so ef=1 stops there;
// ef=3 keeps C as a fallback and reaches E through it.
func TestSearchLayerEscapesLocalMinimum(t *testing.T) {
	h := lineGraph(
		[]float32{5, 3, 4, 6, 1},
		[][2]uint32{{0, 1}, {1, 2}, {1, 3}, {2, 4}, {3, 4}},
	)
	q := []float32{0}

	cases := []struct {
		ef   int
		want []uint32
	}{
		{1, []uint32{1}},
		{3, []uint32{4, 1, 2}},
	}
	for _, c := range cases {
		got := nodeIDs(h.searchLayer(q, []uint32{0}, c.ef, 0))
		if !slices.Equal(got, c.want) {
			t.Errorf("ef=%d: got nodes %v, want %v", c.ef, got, c.want)
		}
	}
}

func TestSearchLayerResultsSorted(t *testing.T) {
	pts := []float32{1, 2, 4, 6, 9, 12, 14}
	h := lineGraph(pts, [][2]uint32{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}})

	got := h.searchLayer([]float32{8}, []uint32{5}, 10, 0)
	if len(got) != len(pts) {
		t.Fatalf("got %d results, want %d", len(got), len(pts))
	}
	if !slices.IsSortedFunc(got, func(a, b candidate) int { return cmp.Compare(a.dist, b.dist) }) {
		t.Errorf("results not nearest-first: %v", got)
	}
}

func TestSelectNeighborsSimple(t *testing.T) {
	cands := []candidate{{7, 1}, {3, 2}, {9, 5}, {1, 8}}

	cases := []struct {
		m    int
		want []uint32
	}{
		{2, []uint32{7, 3}},
		{10, []uint32{7, 3, 9, 1}},
	}
	for _, c := range cases {
		got := nodeIDs(selectNeighborsSimple(cands, c.m))
		if !slices.Equal(got, c.want) {
			t.Errorf("m=%d: got %v, want %v", c.m, got, c.want)
		}
	}
}

// only vec() works on this one
func pointIndex(pts ...[2]float32) *Index {
	h := New(2, Config{M: 4, EfConstruction: 10})
	for _, p := range pts {
		h.vecs = append(h.vecs, p[0], p[1])
	}
	return h
}

// A(2,0) C(0,-3) B(3,1) D(-4,0), base (0,0). B is closer to A (2) than to the base (10),
// so it is dropped; Alg 3 would return [A C B].
func TestSelectNeighborsHeuristicDiverse(t *testing.T) {
	h := pointIndex([2]float32{2, 0}, [2]float32{0, -3}, [2]float32{3, 1}, [2]float32{-4, 0})
	cands := []candidate{{0, 4}, {1, 9}, {2, 10}, {3, 16}}

	got := nodeIDs(h.selectNeighborsHeuristic(cands, 3))
	if want := []uint32{0, 1, 3}; !slices.Equal(got, want) {
		t.Errorf("got %v, want %v (A C D)", got, want)
	}
}

// Once one point of a far cluster is kept, the rest are closer to it than to the base,
// so the heuristic returns fewer than m.
func TestSelectNeighborsHeuristicCluster(t *testing.T) {
	h := pointIndex([2]float32{10, 0}, [2]float32{10, 1}, [2]float32{11, 0})
	cands := []candidate{{0, 100}, {1, 101}, {2, 121}}

	got := nodeIDs(h.selectNeighborsHeuristic(cands, 3))
	if want := []uint32{0}; !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func buildRandom(t *testing.T, n, dim int) (*Index, [][]float32) {
	t.Helper()
	rng := rand.New(rand.NewSource(42))
	h := New(dim, Config{M: 8, EfConstruction: 100, Seed: 1})
	vecs := randomVecs(rng, n, dim)
	for i, v := range vecs {
		if err := h.Add(uint64(i), v); err != nil {
			t.Fatalf("Add(%d): %v", i, err)
		}
	}
	return h, vecs
}

func randomVecs(rng *rand.Rand, n, dim int) [][]float32 {
	vecs := make([][]float32, n)
	for i := range vecs {
		vecs[i] = make([]float32, dim)
		for j := range vecs[i] {
			vecs[i][j] = rng.Float32()
		}
	}
	return vecs
}

// Violations here don't crash; they only lower recall.
func TestAddGraphInvariants(t *testing.T) {
	h, _ := buildRandom(t, 2000, 16)

	if got := h.nodes[h.entry].level; got != h.maxLevel {
		t.Errorf("entry node is on level %d, maxLevel is %d", got, h.maxLevel)
	}

	for i, n := range h.nodes {
		if len(n.neighbors) != n.level+1 {
			t.Fatalf("node %d: %d neighbour lists, want level+1 = %d", i, len(n.neighbors), n.level+1)
		}
		for layer, links := range n.neighbors {
			limit := h.m
			if layer == 0 {
				limit = h.mMax0
			}
			if len(links) > limit {
				t.Errorf("node %d layer %d: %d links, cap %d", i, layer, len(links), limit)
			}
			seen := make(map[uint32]bool, len(links))
			for _, nb := range links {
				switch {
				case nb == uint32(i):
					t.Errorf("node %d layer %d: links to itself", i, layer)
				case int(nb) >= len(h.nodes):
					t.Errorf("node %d layer %d: link to missing node %d", i, layer, nb)
				case h.nodes[nb].level < layer:
					t.Errorf("node %d layer %d: neighbour %d only reaches level %d", i, layer, nb, h.nodes[nb].level)
				case seen[nb]:
					t.Errorf("node %d layer %d: duplicate link to %d", i, layer, nb)
				}
				seen[nb] = true
			}
		}
	}
}

func TestAddCopiesVector(t *testing.T) {
	h := New(3, Config{M: 4, EfConstruction: 10})
	v := []float32{1, 2, 3}
	if err := h.Add(7, v); err != nil {
		t.Fatal(err)
	}

	v[0] = 99
	if got := h.vec(0); !slices.Equal(got, []float32{1, 2, 3}) {
		t.Errorf("vec(0) = %v after caller changed its slice, want [1 2 3]", got)
	}
}

func TestAddWrongDim(t *testing.T) {
	h := New(3, Config{M: 4, EfConstruction: 10})
	err := h.Add(1, []float32{1, 2})
	if !errors.Is(err, errs.ErrDimMismatch) {
		t.Errorf("Add with 2 dims into a 3-dim index: err = %v, want ErrDimMismatch", err)
	}
	if h.Len() != 0 {
		t.Errorf("failed Add still stored a node: Len() = %d", h.Len())
	}
}

func TestSearchErrors(t *testing.T) {
	h := New(3, Config{M: 4, EfConstruction: 10})

	got, err := h.Search([]float32{1, 2, 3}, 5, 10)
	if err != nil || len(got) != 0 {
		t.Errorf("empty index: got %v, %v; want no results, nil", got, err)
	}

	cases := []struct {
		name string
		q    []float32
		k    int
		want error
	}{
		{"wrong dim", []float32{1, 2}, 5, errs.ErrDimMismatch},
		{"k=0", []float32{1, 2, 3}, 0, errs.ErrInvalidArgument},
	}
	for _, c := range cases {
		if _, err := h.Search(c.q, c.k, 10); !errors.Is(err, c.want) {
			t.Errorf("%s: err = %v, want %v", c.name, err, c.want)
		}
	}
}

func TestSearchRecallVsFlat(t *testing.T) {
	const n, dim, k, ef, queries = 2000, 16, 10, 64, 50
	h, vecs := buildRandom(t, n, dim)

	oracle := flat.New(dim)
	for i, v := range vecs {
		if err := oracle.Add(uint64(i), v); err != nil {
			t.Fatal(err)
		}
	}

	hits := 0
	for _, q := range randomVecs(rand.New(rand.NewSource(7)), queries, dim) {
		got, err := h.Search(q, k, ef)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != k {
			t.Fatalf("got %d results, want %d", len(got), k)
		}
		if !slices.IsSortedFunc(got, func(a, b Result) int { return cmp.Compare(a.Dist, b.Dist) }) {
			t.Fatalf("results not nearest-first: %v", got)
		}

		truth, _ := oracle.Search(q, k)
		trueIDs := make(map[uint64]bool, k)
		for _, r := range truth {
			trueIDs[r.ID] = true
		}
		for _, r := range got {
			if trueIDs[r.ID] {
				hits++
			}
		}
	}

	recall := float64(hits) / (queries * k)
	t.Logf("recall@%d = %.3f (n=%d, dim=%d, ef=%d)", k, recall, n, dim, ef)
	if recall < 0.95 {
		t.Errorf("recall@%d = %.3f, want >= 0.95", k, recall)
	}
}
