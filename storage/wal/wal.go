// Write-ahead log. Append returns only after fsync.
//
//	[len u32][crc u32][type u8][payload]
//
// little-endian. len = payload length, crc = CRC32-C over type+payload.
package wal

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"okapi/internal/errs"
	"os"
)

const (
	lenOff     = 0
	crcOff     = lenOff + 4
	typeOff    = crcOff + 4
	headerSize = typeOff + 1
)

var crcTable = crc32.MakeTable(crc32.Castagnoli)

type Record struct {
	Type    byte
	Payload []byte
}

type WAL struct {
	f *os.File
}

func Open(path string) (*WAL, error) {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return nil, fmt.Errorf("wal: open: %w", err)
	}
	return &WAL{f: f}, nil
}

func (w *WAL) Append(typ byte, payload []byte) error {
	if len(payload) > math.MaxUint32 {
		return fmt.Errorf("wal: %w: payload is %d bytes", errs.ErrInvalidArgument, len(payload))
	}

	if _, err := w.f.Write(encode(typ, payload)); err != nil {
		return fmt.Errorf("wal: append: %w", err)
	}
	if err := w.f.Sync(); err != nil {
		return fmt.Errorf("wal: sync: %w", err)
	}
	return nil
}

func (w *WAL) Close() error {
	return w.f.Close()
}

// A torn tail (crash mid-write, or zeros after the last record) isn't an
// error: ReadAll stops there and end is where the good records stop.
//
// Truncate to end before appending again. Bad data with real records
// after it is ErrCorrupt.
func ReadAll(path string) (recs []Record, end int64, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("wal: open: %w", err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, 0, fmt.Errorf("wal: stat: %w", err)
	}
	size := info.Size()
	r := bufio.NewReader(f)

	var offset int64 // start of the current record
	header := make([]byte, headerSize)
	for {
		left := size - offset
		if left < headerSize {
			return recs, offset, nil
		}

		if _, err := io.ReadFull(r, header); err != nil {
			return recs, offset, fmt.Errorf("wal: record %d header: %w", len(recs), err)
		}
		n, want, typ := decodeHeader(header)

		// n can't be more than what's left. also stops a garbage n reaching make()
		if int64(n) > left-headerSize {
			return recs, offset, nil
		}

		payload := make([]byte, n)
		if _, err := io.ReadFull(r, payload); err != nil {
			return recs, offset, fmt.Errorf("wal: record %d payload: %w", len(recs), err)
		}

		if got := makeCrc(typ, payload); got != want {
			rest, err := io.ReadAll(r)
			if err != nil {
				return recs, offset, fmt.Errorf("wal: record %d: %w", len(recs), err)
			}
			if allZero(rest) {
				return recs, offset, nil // torn, or zeros after the end
			}
			return recs, offset, fmt.Errorf("wal: record %d: %w: crc %08x, want %08x", len(recs), errs.ErrCorrupt, got, want)
		}

		recs = append(recs, Record{Type: typ, Payload: payload})
		offset += headerSize + int64(n)
	}
}

func allZero(b []byte) bool {
	for _, c := range b {
		if c != 0 {
			return false
		}
	}
	return true
}

// one buffer, so the record goes out in a single Write
func encode(typ byte, payload []byte) []byte {
	buf := make([]byte, 0, headerSize+len(payload))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(payload)))
	buf = binary.LittleEndian.AppendUint32(buf, makeCrc(typ, payload))
	buf = append(buf, typ)
	return append(buf, payload...)
}

func decodeHeader(h []byte) (n, crc uint32, typ byte) {
	return binary.LittleEndian.Uint32(h[lenOff:crcOff]),
		binary.LittleEndian.Uint32(h[crcOff:typeOff]),
		h[typeOff]
}

func makeCrc(typ byte, payload []byte) uint32 {
	crc := crc32.Update(0, crcTable, []byte{typ})
	return crc32.Update(crc, crcTable, payload)
}
