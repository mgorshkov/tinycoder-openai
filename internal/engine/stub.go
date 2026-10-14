package engine

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// Stub is a deterministic in-process Engine used for development, tests and
// deployments built without the native inference library. It does not run a
// real model: it echoes a short, well-formed response derived from the prompt
// so the whole HTTP surface (registry, streaming, chat templates, accounting)
// can be exercised.
type Stub struct {
	modelID string
	loaded  bool
	info    ModelInfo
}

// NewStub returns a stub engine bound to modelID.
func NewStub(modelID string) *Stub {
	return &Stub{modelID: modelID}
}

// Load marks the stub loaded and fabricates plausible model info.
func (s *Stub) Load(ctx context.Context, opts LoadOptions) error {
	s.loaded = true
	seq := opts.MaxSeqLen
	if seq == 0 {
		seq = 8192
	}
	s.info = ModelInfo{
		ID:            s.modelID,
		Name:          s.modelID,
		Architecture:  "stub",
		NumLayers:     28,
		HiddenSize:    1536,
		NumHeads:      12,
		NumKVHeads:    2,
		VocabSize:     151936,
		MaxSeqLen:     seq,
		ContextLength: seq,
	}
	return nil
}

// Unload clears the stub state.
func (s *Stub) Unload() error { s.loaded = false; s.info = ModelInfo{}; return nil }

// Loaded reports whether Load has been called.
func (s *Stub) Loaded() bool { return s.loaded }

// Info returns fabricated metadata.
func (s *Stub) Info() ModelInfo { return s.info }

// Capabilities reports the stub's feature set.
func (s *Stub) Capabilities() Capabilities {
	return Capabilities{Native: false, Streaming: true, Chat: true}
}

// Generate emits a deterministic token stream. Words of a canned reply are
// streamed one at a time with small delays so streaming clients can be tested.
func (s *Stub) Generate(ctx context.Context, req GenerateRequest, onToken TokenFunc) (Usage, error) {
	if !s.loaded {
		return Usage{}, ErrNotLoaded
	}
	promptPreview := strings.TrimSpace(req.Prompt)
	if len(req.Messages) > 0 {
		promptPreview = strings.TrimSpace(req.Messages[len(req.Messages)-1].Content)
	}
	if len(promptPreview) > 80 {
		promptPreview = promptPreview[:80]
	}
	reply := fmt.Sprintf("[stub:%s] Received %d message(s). Last input: %q", s.modelID, len(req.Messages)+1, promptPreview)

	promptTokens := (len(promptPreview) + 3) / 4
	if promptTokens < 1 {
		promptTokens = 1
	}
	var usage Usage
	usage.PromptTokens = promptTokens

	words := strings.Fields(reply)
	if req.MaxTokens > 0 && len(words) > req.MaxTokens {
		words = words[:req.MaxTokens]
	}
	for i, w := range words {
		select {
		case <-ctx.Done():
			return usage, ErrContextCanceled
		default:
		}
		text := w
		if i < len(words)-1 {
			text += " "
		}
		usage.CompletionTokens++
		if onToken != nil && !onToken(Token{ID: int32(i), Text: text}) {
			break
		}
		// Small pacing delay so SSE streams chunk realistically.
		select {
		case <-ctx.Done():
			return usage, ErrContextCanceled
		case <-time.After(10 * time.Millisecond):
		}
	}
	usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	return usage, nil
}
