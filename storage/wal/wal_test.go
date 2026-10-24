package wal

import (
	"bytes"
	"path/filepath"
	"testing"
)

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
	got, err := ReadAll(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d records, want %d", len(got), len(want))
	}
	for i := range want {
		// bytes.Equal treats nil and empty as equal; the reader returns []byte{}.
		if got[i].Type != want[i].Type || !bytes.Equal(got[i].Payload, want[i].Payload) {
			t.Errorf("record %d: got {%d %q}, want {%d %q}", i, got[i].Type, got[i].Payload, want[i].Type, want[i].Payload)
		}
	}
}

// Write 3 records (one with an empty payload), close, read back, compare.
func TestAppendReadAll(t *testing.T) {
	path := filepath.Join(t.TempDir(), "wal.log")
	recs := []Record{
		{Type: 1, Payload: []byte("hi")},
		{Type: 2, Payload: nil},
		{Type: 1, Payload: []byte("okapi")},
	}
	appendAll(t, path, recs)
	checkRecords(t, path, recs)
}

// Reopen an existing WAL, append a 4th record, and expect all 4 back.
func TestReopenAppends(t *testing.T) {
	path := filepath.Join(t.TempDir(), "wal.log")
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
