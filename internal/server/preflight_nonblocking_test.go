package server

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mgorshkov/tinycoder-openai/internal/engine"
)

// busyEngine holds the first Generate call inside the engine until released and
// refuses every later call with ErrContextLength. It lets a test keep a warm
// model busy while a second streaming request arrives.
type busyEngine struct {
	*windowEngine
	entered chan struct{}
	release chan struct{}
	calls   atomic.Int32
}

func (e *busyEngine) Generate(ctx context.Context, req engine.GenerateRequest, onToken engine.TokenFunc) (engine.Usage, error) {
	if e.calls.Add(1) == 1 {
		close(e.entered)
		<-e.release
		if onToken != nil {
			onToken(engine.Token{Text: "first"})
		}
		return engine.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2}, nil
	}
	return e.windowEngine.Generate(ctx, req, onToken)
}

// newBusyPreflightServer builds a server whose "alpha" model has a 1024-token
// window and a busyEngine that a test can hold inside a first generation.
func newBusyPreflightServer(t *testing.T) (*httptest.Server, *busyEngine, *Manager) {
	t.Helper()
	refusal := fmt.Errorf("%w: prompt is too long: 2000 tokens > 1024 maximum", engine.ErrContextLength)
	cfg := testConfig()
	for i := range cfg.Models {
		if cfg.Models[i].ID == "alpha" {
			cfg.Models[i].ContextLength = 1024
		}
	}
	eng := &busyEngine{
		windowEngine: &windowEngine{Stub: engine.NewStub("alpha"), genErr: refusal},
		entered:      make(chan struct{}),
		release:      make(chan struct{}),
	}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	mgr := NewManager(cfg, func(id string) engine.Engine {
		if id == "alpha" {
			return eng
		}
		return engine.NewStub(id)
	}, log)
	ts := httptest.NewServer(New(cfg, mgr, log).Handler())
	t.Cleanup(ts.Close)
	return ts, eng, mgr
}

// TestBusyWarmStreamPreflightDoesNotBlock pins the non-blocking streaming
// preflight: a streaming request to a warm-but-busy model must open its SSE
// stream immediately instead of waiting behind the first generation for the
// model lock. The second request cannot be refused before the stream opens (the
// prompt count is unavailable while the model is busy), so the refusal must
// surface as an SSE error event with the context code after the first
// generation releases the model.
func TestBusyWarmStreamPreflightDoesNotBlock(t *testing.T) {
	ts, eng, mgr := newBusyPreflightServer(t)
	warmModel(t, mgr)

	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(eng.release) }) }
	defer release()

	const promptDeadline = 2 * time.Second

	// First streaming request: enters the engine and blocks there, holding the
	// model lock for the whole generation.
	firstDone := make(chan error, 1)
	go func() {
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json",
			strings.NewReader(`{"model":"alpha","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
		if err != nil {
			firstDone <- err
			return
		}
		defer resp.Body.Close()
		_, err = io.Copy(io.Discard, resp.Body)
		firstDone <- err
	}()

	select {
	case <-eng.entered:
	case <-time.After(promptDeadline):
		t.Fatal("first streaming request never reached the engine")
	}

	// Second streaming request: over-long prompt against the busy model.
	long := strings.Repeat("x", 2000)
	secondBody := fmt.Sprintf(`{"model":"alpha","stream":true,"messages":[{"role":"user","content":%q}]}`, long)

	type secondResult struct {
		resp  *http.Response
		br    *bufio.Reader
		first string
		err   error
	}
	secondCh := make(chan secondResult, 1)
	go func() {
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(secondBody))
		if err != nil {
			secondCh <- secondResult{err: err}
			return
		}
		br := bufio.NewReader(resp.Body)
		line, err := br.ReadString('\n')
		if err != nil {
			resp.Body.Close()
			secondCh <- secondResult{err: err}
			return
		}
		secondCh <- secondResult{resp: resp, br: br, first: line}
	}()

	var second secondResult
	select {
	case second = <-secondCh:
	case <-time.After(promptDeadline):
		t.Fatal("second streaming request received no headers/first bytes within 2s while the model was busy; the streaming preflight must not block on the model lock")
	}
	if second.err != nil {
		t.Fatalf("second streaming request: %v", second.err)
	}
	defer second.resp.Body.Close()

	if second.resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(second.br)
		t.Fatalf("second status = %d, want 200 (SSE headers must be written first); body=%s", second.resp.StatusCode, raw)
	}

	// The first bytes must be the assistant role chunk, flushed before the
	// second generation waits for the busy model.
	var role chatChunk
	if err := json.Unmarshal([]byte(strings.TrimSpace(strings.TrimPrefix(second.first, "data: "))), &role); err != nil {
		t.Fatalf("first SSE frame %q is not a chat chunk: %v", second.first, err)
	}
	if len(role.Choices) == 0 || role.Choices[0].Delta.Role != "assistant" {
		t.Fatalf("first SSE frame is not the assistant role chunk: %q", second.first)
	}

	// Those bytes must have arrived while the first generation still held the
	// model.
	select {
	case err := <-firstDone:
		t.Fatalf("first generation finished early (%v); it should still be holding the model", err)
	default:
	}

	release()
	select {
	case err := <-firstDone:
		if err != nil {
			t.Fatalf("first streaming request: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("first streaming request did not finish after release")
	}

	var sawCode, sawDone bool
	for _, ev := range readTestSSE(t, second.br) {
		if ev.Data == "[DONE]" {
			sawDone = true
			continue
		}
		if !strings.Contains(ev.Data, `"error"`) {
			continue
		}
		var eb errorBody
		if err := json.Unmarshal([]byte(ev.Data), &eb); err != nil {
			t.Fatalf("bad SSE error frame %q: %v", ev.Data, err)
		}
		if eb.Error.Code != "context_length_exceeded" || eb.Error.Type != "invalid_request_error" {
			t.Fatalf("unexpected SSE error payload: %+v", eb.Error)
		}
		sawCode = true
	}
	if !sawCode {
		t.Fatal("second stream did not surface context_length_exceeded after the model was released")
	}
	if !sawDone {
		t.Fatal("second stream did not terminate with [DONE]")
	}
	// windowEngine.generateCalls counts only calls delegated to it: the blocking
	// first call bypasses it, the refusing second call is counted.
	if got := eng.generateCalls.Load(); got != 1 {
		t.Fatalf("engine Generate calls through windowEngine = %d, want 1 (the refusing second call)", got)
	}
}
