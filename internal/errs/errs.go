// Package errs defines the error kinds shared across Okapi.
// Wrap them with context via fmt.Errorf("...: %w", ...) and check with errors.Is.
// Add a kind only when a caller needs to tell it apart.
package errs

import "errors"

var (
	ErrDimMismatch = errors.New("dimension mismatch")

	ErrInvalidArgument = errors.New("invalid argument")

	ErrNotFound = errors.New("not found")

	ErrCorrupt = errors.New("corrupt data")
)
