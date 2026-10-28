package hnsw

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"syscall"
	"unsafe"
)

// mapVectors maps vectors.bin read-only and returns it as []float32 without copying.
// The OS loads a 4 KB page the first time a search touches it.
// unmap must be called once nothing uses the slice anymore.
func mapVectors(path string, count int) (vecs []float32, unmap func() error, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("hnsw: load: %w", err)
	}
	defer f.Close() // the mapping stays valid after the file is closed

	info, err := f.Stat()
	if err != nil {
		return nil, nil, fmt.Errorf("hnsw: load: %w", err)
	}
	if info.Size() != int64(count)*4 {
		return nil, nil, corrupt("%s is %d bytes, want %d", filepath.Base(path), info.Size(), count*4)
	}
	if count == 0 {
		return nil, func() error { return nil }, nil // mmap of 0 bytes is an error
	}

	b, err := syscall.Mmap(int(f.Fd()), 0, count*4, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, fmt.Errorf("hnsw: mmap %s: %w", filepath.Base(path), err)
	}
	unmap = func() error { return syscall.Munmap(b) }

	// the file is little-endian; on a little-endian CPU (x86, ARM) those bytes
	// already are float32s, so just view them as such
	if binary.NativeEndian.Uint16([]byte{1, 0}) == 1 {
		return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), count), unmap, nil
	}

	// big-endian CPU: decode into a normal slice and drop the mapping
	vecs = make([]float32, count)
	for i := range vecs {
		vecs[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return vecs, func() error { return nil }, unmap()
}

// Close unmaps a loaded index's vectors; don't search it afterwards.
// No-op for an index built in memory.
func (h *Index) Close() error {
	if h.unmap == nil {
		return nil
	}
	err := h.unmap()
	h.unmap, h.vecs = nil, nil
	return err
}
