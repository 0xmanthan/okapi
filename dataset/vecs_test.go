package dataset

import (
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"okapi/internal/errs"
)

const siftDir = "../data/siftsmall/"

func skipIfNoData(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(siftDir); err != nil {
		t.Skipf("siftsmall not found at %s; download it first", siftDir)
	}
}

func TestReadFVecsShapes(t *testing.T) {
	skipIfNoData(t)

	cases := []struct {
		file    string
		wantN   int
		wantDim int
	}{
		{"siftsmall_base.fvecs", 10000, 128},
		{"siftsmall_query.fvecs", 100, 128},
		{"siftsmall_learn.fvecs", 25000, 128},
	}

	for _, c := range cases {
		t.Run(c.file, func(t *testing.T) {
			vecs, err := ReadFVecs(siftDir + c.file)
			if err != nil {
				t.Fatal(err)
			}
			if len(vecs) != c.wantN {
				t.Fatalf("got %d vectors, want %d", len(vecs), c.wantN)
			}
			for i, v := range vecs {
				if len(v) != c.wantDim {
					t.Fatalf("vector %d has dim %d, want %d", i, len(v), c.wantDim)
				}
			}
			t.Logf("%d × %d, first values %v", len(vecs), len(vecs[0]), vecs[0][:8])
		})
	}
}

func TestReadFVecsFirstValues(t *testing.T) {
	skipIfNoData(t)

	vecs, err := ReadFVecs(siftDir + "siftsmall_base.fvecs")
	if err != nil {
		t.Fatal(err)
	}
	// from od -t f4 -j 4 -N 32
	want := []float32{0, 16, 35, 5, 32, 31, 14, 10}
	for i, w := range want {
		if vecs[0][i] != w {
			t.Fatalf("vecs[0][%d] = %v, want %v (byte order wrong?)", i, vecs[0][i], w)
		}
	}
}

func TestReadIVecsGroundTruth(t *testing.T) {
	skipIfNoData(t)

	gt, err := ReadIVecs(siftDir + "siftsmall_groundtruth.ivecs")
	if err != nil {
		t.Fatal(err)
	}
	if len(gt) != 100 {
		t.Fatalf("got %d rows, want 100 (one per query)", len(gt))
	}
	for q, row := range gt {
		if len(row) != 100 {
			t.Fatalf("query %d has %d neighbours, want 100", q, len(row))
		}
		// IDs are positions in the base file.
		for _, id := range row {
			if id < 0 || id >= 10000 {
				t.Fatalf("query %d has id %d, outside [0, 10000)", q, id)
			}
		}
	}
	t.Logf("query 0 top-10 ids: %v", gt[0][:10])
}

func TestReadFVecsMissingFile(t *testing.T) {
	_, err := ReadFVecs("does-not-exist.fvecs")
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("got %v, want an error wrapping os.ErrNotExist", err)
	}
}

func TestReadFVecsTruncated(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.fvecs")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	binary.Write(f, binary.LittleEndian, int32(4))
	binary.Write(f, binary.LittleEndian, []float32{1, 2})
	f.Close()

	_, err = ReadFVecs(path)
	if !errors.Is(err, io.ErrUnexpectedEOF) || !errors.Is(err, errs.ErrCorrupt) {
		t.Fatalf("got %v, want an error wrapping both io.ErrUnexpectedEOF and errs.ErrCorrupt", err)
	}
}

func writeFVecs(t *testing.T, recs ...[]float32) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "x.fvecs")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for _, v := range recs {
		binary.Write(f, binary.LittleEndian, int32(len(v)))
		binary.Write(f, binary.LittleEndian, v)
	}
	return path
}

func TestReadFVecsDimChanges(t *testing.T) {
	path := writeFVecs(t, []float32{1, 2, 3}, []float32{4, 5})

	_, err := ReadFVecs(path)
	if !errors.Is(err, errs.ErrDimMismatch) {
		t.Fatalf("got %v, want errs.ErrDimMismatch", err)
	}
}

func TestReadFVecsNegativeDim(t *testing.T) {
	path := filepath.Join(t.TempDir(), "neg.fvecs")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	binary.Write(f, binary.LittleEndian, int32(-5))
	f.Close()

	_, err = ReadFVecs(path)
	if !errors.Is(err, errs.ErrCorrupt) {
		t.Fatalf("got %v, want errs.ErrCorrupt", err)
	}
}

func TestReadFVecsEmptyFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.fvecs")
	if err := os.WriteFile(path, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	vecs, err := ReadFVecs(path)
	if err != nil {
		t.Fatalf("empty file should not be an error, got %v", err)
	}
	if len(vecs) != 0 {
		t.Fatalf("got %d vectors from an empty file, want 0", len(vecs))
	}
}
