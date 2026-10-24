// Wrap with fmt.Errorf("...: %w", ...), check with errors.Is.
package errs

import "errors"

var (
	ErrDimMismatch = errors.New("dimension mismatch")

	ErrInvalidArgument = errors.New("invalid argument")

	ErrNotFound = errors.New("not found")

	ErrCorrupt = errors.New("corrupt data")
)
