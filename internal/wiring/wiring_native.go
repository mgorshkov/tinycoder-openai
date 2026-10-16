//go:build tinycoder_native

// Package wiring selects the engine implementation at build time. The native
// variant links the tinycoder-inference bridge; the default is the stub.
package wiring

import (
	"github.com/mgorshkov/tinycoder-openai/internal/engine"
	"github.com/mgorshkov/tinycoder-openai/internal/engine/native"
)

// Native reports whether the native engine is compiled in.
const Native = true

// NewFactory returns the native engine factory.
func NewFactory() engine.Factory {
	return func(modelID string) engine.Engine { return native.New(modelID) }
}
