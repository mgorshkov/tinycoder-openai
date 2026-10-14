//go:build !tinycoder_native

package wiring

import "github.com/mgorshkov/tinycoder-openai/internal/engine"

// Native reports whether the native engine is compiled in.
const Native = false

// NewFactory returns the stub engine factory.
func NewFactory() engine.Factory {
	return func(modelID string) engine.Engine { return engine.NewStub(modelID) }
}
