// Package hnsw implements Hierarchical Navigable Small World graphs
// (Malkov & Yashunin, arXiv:1603.09320). Algorithm numbers refer to the paper.
package hnsw

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"slices"

	"okapi/distance"
	"okapi/internal/errs"
)

type Config struct {
	M              int // links per new node per layer; Mmax = M, Mmax0 = 2M
	EfConstruction int
	Seed           int64
}

type Result struct {
	ID   uint64
	Dist float32
}

type candidate struct {
	id   uint32
	dist float32
}

type node struct {
	level     int
	neighbors [][]uint32 // neighbors[layer], for layer 0..level
}

type Index struct {
	dim            int
	m, mMax0       int
	efConstruction int
	mL             float64

	rng   *rand.Rand
	vecs  []float32 // slab: node i is vecs[i*dim : (i+1)*dim]
	ids   []uint64  // node i → external id
	nodes []node

	entry    int32 // -1 when empty
	maxLevel int

	// Visited set for searchLayer: node i is visited in the current search iff
	// visitMark[i] == visitEpoch. Bumping the epoch clears it in O(1).
	// uint64 so the epoch never wraps.
	visitMark  []uint64
	visitEpoch uint64
}

func New(dim int, cfg Config) *Index {
	return &Index{
		dim:            dim,
		m:              cfg.M,
		mMax0:          2 * cfg.M,
		efConstruction: cfg.EfConstruction,
		mL:             1 / math.Log(float64(cfg.M)),
		rng:            rand.New(rand.NewSource(cfg.Seed)),
		entry:          -1,
	}
}

func (h *Index) Len() int { return len(h.nodes) }

func (h *Index) vec(i uint32) []float32 {
	dim := uint32(h.dim)
	return h.vecs[i*dim : (i+1)*dim]
}

func (h *Index) randomLevel() int {
	u := 1 - h.rng.Float64()
	return int(math.Floor(-math.Log(u) * h.mL))
}

// newVisitEpoch starts a fresh visited set, growing visitMark to cover every node.
func (h *Index) newVisitEpoch() {
	if n := len(h.nodes); len(h.visitMark) < n {
		h.visitMark = append(h.visitMark, make([]uint64, n-len(h.visitMark))...)
	}
	h.visitEpoch++
}

func (h *Index) searchLayer(q []float32, eps []uint32, ef, layer int) []candidate {
	candidates := &minHeap{}
	results := &maxHeap{}

	h.newVisitEpoch()

	for _, ep := range eps {
		d, _ := distance.L2Squared(q, h.vec(ep))
		h.visitMark[ep] = h.visitEpoch
		heap.Push(candidates, candidate{id: ep, dist: d})
		heap.Push(results, candidate{id: ep, dist: d})
	}
	for results.Len() > ef {
		heap.Pop(results)
	}

	for candidates.Len() > 0 {
		c := heap.Pop(candidates).(candidate)
		worst := (*results)[0]
		if c.dist > worst.dist {
			break
		}

		for _, e := range h.nodes[c.id].neighbors[layer] {
			if h.visitMark[e] == h.visitEpoch {
				continue
			}
			h.visitMark[e] = h.visitEpoch

			d, _ := distance.L2Squared(q, h.vec(e))
			if d < (*results)[0].dist || results.Len() < ef {
				heap.Push(candidates, candidate{id: e, dist: d})
				heap.Push(results, candidate{id: e, dist: d})
				if results.Len() > ef {
					heap.Pop(results)
				}
			}
		}
	}

	out := make([]candidate, results.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(results).(candidate)
	}
	return out
}

// selectNeighborsSimple is Alg 3: the m nearest of cands, nearest first. cands may be unsorted.
func selectNeighborsSimple(cands []candidate, m int) []candidate {
	nearest := minHeap(slices.Clone(cands))
	heap.Init(&nearest)

	out := make([]candidate, 0, min(len(cands), m))
	for range cap(out) {
		out = append(out, heap.Pop(&nearest).(candidate))
	}
	return out
}

// selectNeighborsHeuristic is Alg 4 with extendCandidates and keepPrunedConnections off.
func (h *Index) selectNeighborsHeuristic(cands []candidate, m int) []candidate {
	nearest := minHeap(slices.Clone(cands))
	heap.Init(&nearest)

	kept := make([]candidate, 0, m)
	for nearest.Len() > 0 && len(kept) < m {
		e := heap.Pop(&nearest).(candidate)

		diverse := true
		for _, r := range kept {
			dToKept, _ := distance.L2Squared(h.vec(e.id), h.vec(r.id))
			if dToKept < e.dist {
				diverse = false
				break
			}
		}
		if diverse {
			kept = append(kept, e)
		}
	}
	return kept
}

// Add is Alg 1 with Alg 4 neighbour selection.
func (h *Index) Add(id uint64, v []float32) error {
	if len(v) != h.dim {
		return fmt.Errorf("hnsw: %w: got %d, want %d", errs.ErrDimMismatch, len(v), h.dim)
	}
	// NaN compares false with everything, which breaks the heaps' ordering.
	for i, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return fmt.Errorf("hnsw: %w: v[%d] = %v", errs.ErrInvalidArgument, i, x)
		}
	}
	if len(h.nodes) == math.MaxUint32 {
		return fmt.Errorf("hnsw: %w: index full (%d nodes)", errs.ErrInvalidArgument, len(h.nodes))
	}

	level := h.randomLevel()
	newID := uint32(len(h.nodes))

	h.vecs = append(h.vecs, v...) // append copies, so the caller's slice isn't shared
	h.ids = append(h.ids, id)
	h.nodes = append(h.nodes, node{
		level:     level,
		neighbors: make([][]uint32, level+1),
	})

	if h.entry == -1 {
		h.entry = int32(newID)
		h.maxLevel = level
		return nil
	}

	ep := uint32(h.entry)
	topLevel := h.maxLevel

	// Phase 1: greedy descent through layers above the new node's level.
	for layer := topLevel; layer > level; layer-- {
		ep = h.searchLayer(v, []uint32{ep}, 1, layer)[0].id
	}

	// Phase 2: on each of the new node's layers, link to M neighbours and shrink any that overflow.
	for layer := min(topLevel, level); layer >= 0; layer-- {
		found := h.searchLayer(v, []uint32{ep}, h.efConstruction, layer)
		chosen := h.selectNeighborsHeuristic(found, h.m)

		for _, nb := range chosen {
			h.nodes[newID].neighbors[layer] = append(h.nodes[newID].neighbors[layer], nb.id)
			h.nodes[nb.id].neighbors[layer] = append(h.nodes[nb.id].neighbors[layer], newID)
		}

		limit := h.m
		if layer == 0 {
			limit = h.mMax0
		}
		for _, nb := range chosen {
			links := h.nodes[nb.id].neighbors[layer]
			if len(links) <= limit {
				continue
			}
			// Re-select nb's links by distance to nb. Dropped links stay on the other side.
			linkCands := make([]candidate, 0, len(links))
			for _, other := range links {
				d, _ := distance.L2Squared(h.vec(nb.id), h.vec(other))
				linkCands = append(linkCands, candidate{id: other, dist: d})
			}
			kept := h.selectNeighborsHeuristic(linkCands, limit)
			newLinks := make([]uint32, 0, len(kept))
			for _, k := range kept {
				newLinks = append(newLinks, k.id)
			}
			h.nodes[nb.id].neighbors[layer] = newLinks
		}

		ep = found[0].id
	}

	if level > topLevel {
		h.entry = int32(newID)
		h.maxLevel = level
	}
	return nil
}

// Search is Alg 5: the k nearest stored vectors to q, nearest first.
// ef is the layer-0 beam width; it's raised to k if smaller.
func (h *Index) Search(q []float32, k, ef int) ([]Result, error) {
	if k <= 0 {
		return nil, fmt.Errorf("hnsw: %w: k=%d", errs.ErrInvalidArgument, k)
	}
	if len(q) != h.dim {
		return nil, fmt.Errorf("hnsw: %w: got %d, want %d", errs.ErrDimMismatch, len(q), h.dim)
	}
	if h.entry == -1 {
		return nil, nil
	}
	ef = max(ef, k)

	ep := uint32(h.entry)
	for layer := h.maxLevel; layer > 0; layer-- {
		ep = h.searchLayer(q, []uint32{ep}, 1, layer)[0].id
	}
	found := h.searchLayer(q, []uint32{ep}, ef, 0)

	out := make([]Result, min(k, len(found)))
	for i := range out {
		out[i] = Result{ID: h.ids[found[i].id], Dist: found[i].dist}
	}
	return out, nil
}
