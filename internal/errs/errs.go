// Package errs holds the error kinds shared across Okapi.
//
// Every package wraps one of these with its own context:
//
//	return fmt.Errorf("distance: %w: %d vs %d", errs.ErrDimMismatch, len(a), len(b))
//
// and callers check the kind, not the message:
//
//	if errors.Is(err, errs.ErrDimMismatch) { ... }
//
// The API layer (Phase 5) maps each kind to one HTTP/gRPC status code.
// Add a new kind only when some caller needs to tell it apart.
package errs

import "errors"

var (
	// ErrDimMismatch: a vector's length doesn't match what was expected. → 400 / InvalidArgument
	ErrDimMismatch = errors.New("dimension mismatch")

	// ErrInvalidArgument: bad input that isn't a dimension problem (k <= 0, NaN, empty id). → 400 / InvalidArgument
	ErrInvalidArgument = errors.New("invalid argument")

	// ErrNotFound: the requested id or collection doesn't exist. → 404 / NotFound
	ErrNotFound = errors.New("not found")

	// ErrCorrupt: data on disk doesn't match its format (bad header, wrong size, CRC mismatch). → 500 / DataLoss
	ErrCorrupt = errors.New("corrupt data")
)
