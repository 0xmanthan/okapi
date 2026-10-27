// A mutable segment: WAL on disk + HNSW in memory.
package segment

import (
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"

	"okapi/index/hnsw"
	"okapi/internal/errs"
	"okapi/storage/wal"
)

type Segment struct {
	dim int
	wal *wal.WAL
	idx *hnsw.Index
}

func Open(dir string, dim int, cfg hnsw.Config) (*Segment, error) {
	path := filepath.Join(dir, "wal.log")

	recs, end, err := wal.ReadAll(path)
	switch {
	case errors.Is(err, fs.ErrNotExist):
	case err != nil:
		return nil, fmt.Errorf("segment: open: %w", err)
	default:
		if err := os.Truncate(path, end); err != nil {
			return nil, fmt.Errorf("segment: truncate: %w", err)
		}
	}

	idx := hnsw.New(dim, cfg)
	for i, rec := range recs {
		if rec.Type != TypeUpsert {
			continue
		}
		id, vec, err := decodeUpsert(rec.Payload)
		if err != nil {
			return nil, fmt.Errorf("segment: replay record %d: %w", i, err)
		}
		if err := idx.Add(id, vec); err != nil {
			return nil, fmt.Errorf("segment: replay record %d: %w", i, err)
		}
	}

	w, err := wal.Open(path)
	if err != nil {
		return nil, fmt.Errorf("segment: %w", err)
	}
	return &Segment{dim: dim, wal: w, idx: idx}, nil
}

func (s *Segment) Upsert(id uint64, vec []float32) error {
	if len(vec) != s.dim {
		return fmt.Errorf("segment: %w: got %d, want %d", errs.ErrDimMismatch, len(vec), s.dim)
	}

	for i, x := range vec {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return fmt.Errorf("segment: %w: vec[%d] = %v", errs.ErrInvalidArgument, i, x)
		}
	}

	if err := s.wal.Append(TypeUpsert, encodeUpsert(id, vec)); err != nil {
		return fmt.Errorf("segment: upsert %d: %w", id, err)
	}

	if err := s.idx.Add(id, vec); err != nil {
		return fmt.Errorf("segment: upsert %d: %w", id, err)
	}
	return nil
}

func (s *Segment) Search(q []float32, k, ef int) ([]hnsw.Result, error) {
	return s.idx.Search(q, k, ef)
}

func (s *Segment) Close() error {
	return s.wal.Close()
}
