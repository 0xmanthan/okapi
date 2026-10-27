// A mutable segment: WAL on disk + HNSW in memory.
package segment

import (
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"

	"okapi/index/hnsw"
	"okapi/internal/errs"
	"okapi/storage/wal"
)

type Segment struct {
	dim     int
	cfg     hnsw.Config
	walPath string
	wal     *wal.WAL
	idx     *hnsw.Index
}

// Open replays the WAL at path into a fresh index. Which WAL is live comes
// from the manifest; a missing file is a new, empty segment.
func Open(path string, dim int, cfg hnsw.Config) (*Segment, error) {
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
	return &Segment{dim: dim, cfg: cfg, walPath: path, wal: w, idx: idx}, nil
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

func (s *Segment) Len() int { return s.idx.Len() }

func (s *Segment) Search(q []float32, k, ef int) ([]hnsw.Result, error) {
	return s.idx.Search(q, k, ef)
}

// Seal saves the index to segDir and moves writes to a new, empty WAL at nextWAL.
// The old WAL is returned, not deleted: until the manifest lists segDir, recovery
// only knows about this data through the old WAL.
func (s *Segment) Seal(segDir, nextWAL string) (oldWAL string, err error) {
	if _, err := os.Stat(nextWAL); err == nil {
		return "", fmt.Errorf("segment: seal: %w: %s already exists", errs.ErrInvalidArgument, nextWAL)
	}
	if err := s.idx.Save(segDir); err != nil {
		return "", fmt.Errorf("segment: seal: %w", err)
	}
	w, err := wal.Open(nextWAL)
	if err != nil {
		return "", fmt.Errorf("segment: seal: %w", err)
	}

	old, oldPath := s.wal, s.walPath
	s.wal, s.walPath, s.idx = w, nextWAL, hnsw.New(s.dim, s.cfg)

	// everything in the old WAL was fsynced by Append, so a failed Close loses nothing
	if err := old.Close(); err != nil {
		return oldPath, fmt.Errorf("segment: seal: close old wal: %w", err)
	}
	return oldPath, nil
}

func (s *Segment) Close() error {
	return s.wal.Close()
}
