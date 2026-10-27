package segment

import (
	"encoding/binary"
	"fmt"
	"math"
	"okapi/internal/errs"
)

const (
	TypeUpsert byte = 1
	TypeDelete byte = 2
)

// upsert: [id u64][dim x float32]
// delete: [id u64]
func encodeUpsert(id uint64, vec []float32) []byte {

	payload := make([]byte, 0, 8+len(vec)*4)

	payload = binary.LittleEndian.AppendUint64(payload, id)

	for i := range vec {
		payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(vec[i]))
	}

	return payload
}

func BytesToFloat32(b []byte) float32 {
	bits := binary.LittleEndian.Uint32(b)
	return math.Float32frombits(bits)
}

func decodeUpsert(p []byte) (id uint64, vec []float32, err error) {
	n := len(p)

	if n < 8 {
		return 0, nil, fmt.Errorf("segment: %w: invalid length %d", errs.ErrInvalidArgument, n)
	}
	id = binary.LittleEndian.Uint64(p[0:8])

	if (n-8)%4 != 0 {
		return 0, nil, fmt.Errorf("segment: %w: invalid dimension %d", errs.ErrInvalidArgument, n-8)
	}

	for i := range (n - 8) / 4 {
		vec = append(vec, BytesToFloat32(p[i*4+8:i*4+12]))
	}

	return id, vec, err
}
