// Package engine defines the backend-agnostic contract used by the
// tinycoder-openai server to run GGUF inference.
//
// The OpenAPI HTTP layer never talks to C++ directly: it depends only on the
// Engine interface below. The real implementation lives in
// internal/engine/native (CGO + a thin C ABI shim over the tinycoder-inference
// C++ engine), while tests inject a fake. This keeps the HTTP surface fully
// testable without the native toolchain.
package engine

import (
	"context"
	"errors"
)

// Sentinel errors returned by Engine implementations.
var (
	// ErrUnavailable indicates the native inference engine could not be used
	// (binary built without native support, or the shared library is missing).
	ErrUnavailable = errors.New("native inference engine unavailable")
	// ErrNotLoaded indicates the engine has no model loaded.
	ErrNotLoaded = errors.New("model not loaded")
	// ErrContextCanceled wraps StopIteration from a canceled request.
	ErrContextCanceled = errors.New("generation canceled")
)

// Message is a single chat turn.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// LoadOptions configures a single model load.
type LoadOptions struct {
	ModelPath string
	NThreads  int
	MaxSeqLen int
}

// ModelInfo is the runtime metadata reported for a loaded model.
type ModelInfo struct {
	ID            string `json:"id"`
	Name          string `json:"name"`
	Architecture  string `json:"architecture"`
	Quantization  string `json:"quantization,omitempty"`
	NumLayers     int    `json:"num_layers"`
	HiddenSize    int    `json:"hidden_size"`
	NumHeads      int    `json:"num_attention_heads"`
	NumKVHeads    int    `json:"num_kv_heads"`
	VocabSize     int    `json:"vocab_size"`
	MaxSeqLen     int    `json:"max_seq_len"`
	ContextLength int    `json:"context_length"`
	SizeBytes     int64  `json:"size_bytes,omitempty"`
}

// GenerateRequest is a single completion request.
//
// Exactly one of Prompt or Messages should be set. When Messages is non-empty
// it takes precedence and is rendered through the model's chat template.
type GenerateRequest struct {
	Model    string
	Prompt   string
	Messages []Message

	MaxTokens     int
	Temperature   float32
	TopP          float32
	TopK          float32
	RepeatPenalty float32
	RepeatLastN   int
	Seed          uint32
}

// Token is one generated token (id + decoded text).
type Token struct {
	ID   int32  `json:"id"`
	Text string `json:"text"`
}

// Usage reports token accounting for a request.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// GenerateResult is the aggregate result of a Generate call.
type GenerateResult struct {
	Text   string
	Tokens []Token
	Usage  Usage
}

// TokenFunc is invoked for every generated token. Returning false requests an
// early stop; the engine then unwinds cleanly.
type TokenFunc func(Token) bool

// Capabilities describes what an Engine implementation supports. It is exposed
// through the models endpoint so clients can feature-detect.
type Capabilities struct {
	Native    bool `json:"native"`
	Streaming bool `json:"streaming"`
	Chat      bool `json:"chat"`
}

// Engine is the contract every inference backend must satisfy. Implementations
// are not required to be goroutine-safe: the load manager guarantees a single
// caller per loaded model at a time.
type Engine interface {
	// Load loads the GGUF model at opts.ModelPath. It must be idempotent-safe
	// for a fresh instance and return a descriptive error on failure.
	Load(ctx context.Context, opts LoadOptions) error
	// Unload releases the model and all associated memory.
	Unload() error
	// Loaded reports whether a model is currently resident.
	Loaded() bool
	// Info returns runtime metadata for the loaded model (zero value if none).
	Info() ModelInfo
	// Generate streams tokens through onToken (which may be nil) and returns
	// aggregate token usage.
	Generate(ctx context.Context, req GenerateRequest, onToken TokenFunc) (Usage, error)
	// Capabilities reports supported features.
	Capabilities() Capabilities
}

// Factory constructs a fresh, empty Engine for the given model id.
type Factory func(modelID string) Engine
