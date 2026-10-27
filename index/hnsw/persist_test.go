package hnsw

import (
	"errors"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"okapi/internal/errs"
	"okapi/internal/testutil"
)

func TestSaveLoadSameResults(t *testing.T) {
	h, _ := buildRandom(t, 1000, 16)
	dir := testutil.OutDir(t)
	if err := h.Save(dir); err != nil {
		t.Fatal(err)
	}
	got, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(got.vecs, h.vecs) || !slices.Equal(got.ids, h.ids) ||
		!reflect.DeepEqual(got.nodes, h.nodes) || got.entry != h.entry || got.maxLevel != h.maxLevel {
		t.Fatal("loaded index differs from the saved one")
	}

	rng := rand.New(rand.NewSource(7))
	for i, q := range randomVecs(rng, 10, 16) {
		want, _ := h.Search(q, 10, 50)
		res, err := got.Search(q, 10, 50)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(res, want) {
			t.Errorf("query %d: got %v, want %v", i, res, want)
		}
	}
}

func TestSaveLoadEmpty(t *testing.T) {
	dir := testutil.OutDir(t)
	if err := New(4, Config{M: 8, EfConstruction: 50}).Save(dir); err != nil {
		t.Fatal(err)
	}
	h, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if res, err := h.Search([]float32{0, 0, 0, 0}, 5, 10); err != nil || len(res) != 0 {
		t.Fatalf("got %v, %v; want no results", res, err)
	}
}

func TestLoadCorrupt(t *testing.T) {
	h, _ := buildRandom(t, 100, 8)

	for _, tc := range []struct {
		name string
		file string
		edit func([]byte) []byte
	}{
		{"graph cut short", graphFile, func(b []byte) []byte { return b[:len(b)-3] }},
		{"vectors cut short", vectorsFile, func(b []byte) []byte { return b[:len(b)-4] }},
		{"ids extra byte", idsFile, func(b []byte) []byte { return append(b, 0) }},
		{"huge n", graphFile, func(b []byte) []byte { b[12], b[13], b[14], b[15] = 0xff, 0xff, 0xff, 0x7f; return b }},
		{"negative entry", graphFile, func(b []byte) []byte { b[16], b[17], b[18], b[19] = 0xfb, 0xff, 0xff, 0xff; return b }}, // -5
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := testutil.OutDir(t)
			if err := h.Save(dir); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(dir, tc.file)
			b, _ := os.ReadFile(path)
			if err := os.WriteFile(path, tc.edit(b), 0644); err != nil {
				t.Fatal(err)
			}
			if _, err := Load(dir); !errors.Is(err, errs.ErrCorrupt) {
				t.Fatalf("err = %v, want ErrCorrupt", err)
			}
		})
	}
}
