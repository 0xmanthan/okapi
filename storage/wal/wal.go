// Package wal is Okapi's write-ahead log: every write is appended and fsynced
// here before it is acknowledged.
//
// Record layout (little-endian, 9-byte header):
//
//	[len u32][crc u32][type u8][payload, len bytes]
//
// len is the payload length; crc is CRC32-Castagnoli over type + payload.
package wal

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"os"

	"okapi/internal/errs"
)

const headerSize = 9

var crcTable = crc32Table()

type Record struct {
	Type    byte
	Payload []byte
}

type WAL struct {
	f *os.File
}

// Open opens path for appending, creating it if needed.
func Open(path string) (*WAL, error) {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return nil, fmt.Errorf("wal: open: %w", err)
	}
	return &WAL{f: f}, nil
}

// Append writes one record and fsyncs it. The record is durable only once
// Append returns nil.
func (w *WAL) Append(typ byte, payload []byte) error {
	if len(payload) > math.MaxUint32 {
		return fmt.Errorf("wal: %w: payload is %d bytes", errs.ErrInvalidArgument, len(payload))
	}

	crc := crc32.Update(0, crcTable, []byte{typ})
	crc = crc32.Update(crc, crcTable, payload)

	// One buffer so the whole record goes out in a single Write.
	buf := make([]byte, 0, headerSize+len(payload))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(payload)))
	buf = binary.LittleEndian.AppendUint32(buf, crc)
	buf = append(buf, typ)
	buf = append(buf, payload...)

	if _, err := w.f.Write(buf); err != nil {
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

// ReadAll returns every record in the file at path, in write order.
func ReadAll(path string) ([]Record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("wal: open: %w", err)
	}
	defer f.Close()
	r := bufio.NewReader(f)

	var recs []Record
	header := make([]byte, headerSize)
	for {
		// io.EOF here means the file ended exactly on a record boundary.
		if _, err := io.ReadFull(r, header); err != nil {
			if err == io.EOF {
				return recs, nil
			}
			return recs, fmt.Errorf("wal: record %d header: %w", len(recs), err)
		}
		n := binary.LittleEndian.Uint32(header[0:4])
		want := binary.LittleEndian.Uint32(header[4:8])
		typ := header[8]

		payload := make([]byte, n)
		if _, err := io.ReadFull(r, payload); err != nil {
			return recs, fmt.Errorf("wal: record %d payload: %w", len(recs), err)
		}

		got := crc32.Update(0, crcTable, []byte{typ})
		got = crc32.Update(got, crcTable, payload)
		if got != want {
			return recs, fmt.Errorf("wal: record %d: %w: crc %08x, want %08x", len(recs), errs.ErrCorrupt, got, want)
		}
		recs = append(recs, Record{Type: typ, Payload: payload})
	}
}

func crc32Table() *crc32.Table {
	return crc32.MakeTable(crc32.Castagnoli)
}
