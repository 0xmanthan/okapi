// Searching across segments. The manifest-backed Store comes next.
package store

import (
	"cmp"
	"fmt"
	"slices"

	"okapi/index/hnsw"
)

// Searcher is anything with its own index: a sealed *hnsw.Index or the mutable *segment.Segment.
type Searcher interface {
	Search(q []float32, k, ef int) ([]hnsw.Result, error)
}

// Search asks every segment for its k nearest and keeps the best k overall.
// A true neighbour can sit in any segment, so none can be skipped.
func Search(segs []Searcher, q []float32, k, ef int) ([]hnsw.Result, error) {
	var all []hnsw.Result
	for i, s := range segs {
		res, err := s.Search(q, k, ef)
		if err != nil {
			return nil, fmt.Errorf("store: segment %d: %w", i, err)
		}
		all = append(all, res...)
	}
	// ties broken by ID so the order doesn't depend on segment order
	slices.SortFunc(all, func(a, b hnsw.Result) int {
		return cmp.Or(cmp.Compare(a.Dist, b.Dist), cmp.Compare(a.ID, b.ID))
	})
	return all[:min(k, len(all))], nil
}
