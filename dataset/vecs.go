package dataset

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"okapi/internal/errs"
)

// fvecs: [int32 d][d float32s] per vector, little-endian.
func ReadFVecs(path string) ([][]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("dataset: open fvecs %s: %w", path, err)
	}
	defer file.Close()

	r := bufio.NewReader(file)
	var vecs [][]float32
	firstD := int32(-1)

	for {
		d, err := readDim(r, len(vecs), &firstD)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("dataset: %s: %w", path, err)
		}

		v := make([]float32, d)
		if err := binary.Read(r, binary.LittleEndian, v); err != nil {
			return nil, fmt.Errorf("dataset: %s: record %d: read values: %w: %w", path, len(vecs), errs.ErrCorrupt, err)
		}
		vecs = append(vecs, v)
	}
	return vecs, nil
}

// ivecs: same layout, int32 values.
func ReadIVecs(path string) ([][]int32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("dataset: open ivecs %s: %w", path, err)
	}
	defer file.Close()

	r := bufio.NewReader(file)
	var vecs [][]int32
	firstD := int32(-1)

	for {
		d, err := readDim(r, len(vecs), &firstD)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("dataset: %s: %w", path, err)
		}

		v := make([]int32, d)
		if err := binary.Read(r, binary.LittleEndian, v); err != nil {
			return nil, fmt.Errorf("dataset: %s: record %d: read values: %w: %w", path, len(vecs), errs.ErrCorrupt, err)
		}
		vecs = append(vecs, v)
	}
	return vecs, nil
}

// plain io.EOF at a clean end, callers check with ==.
// firstD is -1 until the first record.
func readDim(r io.Reader, record int, firstD *int32) (int32, error) {
	var d int32
	err := binary.Read(r, binary.LittleEndian, &d)
	if err == io.EOF {
		return 0, io.EOF
	}
	if err != nil {
		return 0, fmt.Errorf("record %d: read dim: %w: %w", record, errs.ErrCorrupt, err)
	}
	if d <= 0 {
		return 0, fmt.Errorf("record %d: %w: d=%d", record, errs.ErrCorrupt, d)
	}
	if *firstD == -1 {
		*firstD = d
	} else if d != *firstD {
		return 0, fmt.Errorf("record %d: %w: d=%d, want %d", record, errs.ErrDimMismatch, d, *firstD)
	}
	return d, nil
}
