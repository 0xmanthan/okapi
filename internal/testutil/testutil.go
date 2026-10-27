package testutil

import (
	"os"
	"path/filepath"
	"testing"
)

// OutDir is like t.TempDir, but under testdata/out/<TestName> in the package
// folder, and kept after the run so the files can be opened. It's emptied
// when the test starts.
func OutDir(t testing.TB) string {
	t.Helper()
	dir := filepath.Join("testdata", "out", t.Name())
	if err := os.RemoveAll(dir); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	return dir
}
