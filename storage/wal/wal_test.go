package wal

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"okapi/internal/errs"
	"okapi/internal/testutil"
)

func walPath(t *testing.T) string {
	t.Helper()
	return filepath.Join(testutil.OutDir(t), "wal.log")
}

func appendAll(t *testing.T, path string, recs []Record) {
	t.Helper()
	w, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range recs {
		if err := w.Append(r.Type, r.Payload); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
}

func checkRecords(t *testing.T, path string, want []Record) {
	t.Helper()
	got, _, err := ReadAll(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d records, want %d", len(got), len(want))
	}
	for i := range want {
		// bytes.Equal: nil == []byte{}
		if got[i].Type != want[i].Type || !bytes.Equal(got[i].Payload, want[i].Payload) {
			t.Errorf("record %d: got {%d %q}, want {%d %q}", i, got[i].Type, got[i].Payload, want[i].Type, want[i].Payload)
		}
	}
}

func TestAppendReadAll(t *testing.T) {
	path := walPath(t)
	recs := []Record{
		{Type: 1, Payload: []byte("hi")},
		{Type: 2, Payload: nil},
		{Type: 1, Payload: []byte("okapi")},
	}
	appendAll(t, path, recs)
	checkRecords(t, path, recs)
}

func TestReopenAppends(t *testing.T) {
	path := walPath(t)
	first := []Record{
		{Type: 1, Payload: []byte("hi")},
		{Type: 2, Payload: nil},
		{Type: 1, Payload: []byte("okapi")},
	}
	fourth := Record{Type: 2, Payload: []byte{0x07, 0, 0, 0, 0, 0, 0, 0}}

	appendAll(t, path, first)
	appendAll(t, path, []Record{fourth})
	checkRecords(t, path, append(first, fourth))
}

var threeRecs = []Record{
	{Type: 1, Payload: []byte("hi")},
	{Type: 2, Payload: nil},
	{Type: 1, Payload: []byte("okapi")},
}

// cut the file at every byte inside the last record
func TestTornTail(t *testing.T) {
	path := walPath(t)
	appendAll(t, path, threeRecs)
	full, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lastStart := int64(len(full) - (headerSize + len("okapi")))

	for cut := lastStart; cut < int64(len(full)); cut++ {
		if err := os.WriteFile(path, full[:cut], 0644); err != nil {
			t.Fatal(err)
		}
		recs, end, err := ReadAll(path)
		if err != nil || len(recs) != 2 || end != lastStart {
			t.Fatalf("cut at %d: got %d records, end %d, err %v; want 2, %d, nil", cut, len(recs), end, err, lastStart)
		}
	}
}

// ext4 can leave zeros after a crash
func TestZeroTail(t *testing.T) {
	path := walPath(t)
	appendAll(t, path, threeRecs)
	full, _ := os.ReadFile(path)
	if err := os.WriteFile(path, append(full, make([]byte, 512)...), 0644); err != nil {
		t.Fatal(err)
	}
	recs, end, err := ReadAll(path)
	if err != nil || len(recs) != 3 || end != int64(len(full)) {
		t.Fatalf("got %d records, end %d, err %v; want 3, %d, nil", len(recs), end, err, len(full))
	}
}

func TestCorruptMiddle(t *testing.T) {
	path := walPath(t)
	appendAll(t, path, threeRecs)
	full, _ := os.ReadFile(path)
	full[headerSize] ^= 1 // first payload byte of record 0
	if err := os.WriteFile(path, full, 0644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := ReadAll(path); !errors.Is(err, errs.ErrCorrupt) {
		t.Fatalf("err = %v, want ErrCorrupt", err)
	}
}

func TestTruncateThenAppend(t *testing.T) {
	path := walPath(t)
	appendAll(t, path, threeRecs)
	full, _ := os.ReadFile(path)
	if err := os.WriteFile(path, full[:len(full)-3], 0644); err != nil { // tear record 2
		t.Fatal(err)
	}
	_, end, err := ReadAll(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(path, end); err != nil {
		t.Fatal(err)
	}
	next := Record{Type: 1, Payload: []byte("after")}
	appendAll(t, path, []Record{next})
	checkRecords(t, path, []Record{threeRecs[0], threeRecs[1], next})
}
