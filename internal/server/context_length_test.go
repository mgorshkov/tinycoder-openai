package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mgorshkov/tinycoder-openai/internal/config"
	"github.com/mgorshkov/tinycoder-openai/internal/engine"
)

// windowEngine wraps engine.Stub so a test can control the reported context
// window, the prompt-token count and the Generate outcome independently of the
// real tokenizer.
type windowEngine struct {
	*engine.Stub
	generateCalls atomic.Int32
	// promptTokens, when positive, is the count returned for every request.
	// Zero makes the engine count the request's characters instead (one per
	// byte) so a test can drive the boundary precisely.
	promptTokens int
	// infoWindow, when non-nil, overrides Info().ContextLength.
	infoWindow *int
	// genErr, when non-nil, is returned from Generate (after load).
	genErr error
}

func (e *windowEngine) Info() engine.ModelInfo {
	info := e.Stub.Info()
	if e.infoWindow != nil {
		info.ContextLength = *e.infoWindow
	}
	return info
}

func (e *windowEngine) CountPromptTokens(req engine.GenerateRequest) (int, error) {
	if e.promptTokens > 0 {
		return e.promptTokens, nil
	}
	n := len(req.Prompt)
	for _, m := range req.Messages {
		n += len(m.Content)
	}
	return n, nil
}

func (e *windowEngine) Generate(ctx context.Context, req engine.GenerateRequest, onToken engine.TokenFunc) (engine.Usage, error) {
	e.generateCalls.Add(1)
	if e.genErr != nil {
		return engine.Usage{}, e.genErr
	}
	return e.Stub.Generate(ctx, req, onToken)
}

// newContextLengthServer builds a server whose "alpha" model has the given
// context window and whose engine is the shared windowEngine returned.
func newContextLengthServer(t *testing.T, window int, configure func(*windowEngine)) (*httptest.Server, *windowEngine) {
	t.Helper()
	cfg := testConfig()
	for i := range cfg.Models {
		if cfg.Models[i].ID == "alpha" {
			cfg.Models[i].ContextLength = window
		}
	}
	eng := &windowEngine{Stub: engine.NewStub("alpha")}
	if configure != nil {
		configure(eng)
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
	return ts, eng
}

// assertContextLengthError decodes an OpenAI-style error body and checks the
// context_length_exceeded shape.
func assertContextLengthError(t *testing.T, resp *http.Response) {
	t.Helper()
	if resp.StatusCode != http.StatusBadRequest {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 400; body=%s", resp.StatusCode, body)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Fatalf("Content-Type = %q, want JSON", ct)
	}
	var eb errorBody
	if err := json.NewDecoder(resp.Body).Decode(&eb); err != nil {
		t.Fatal(err)
	}
	if eb.Error.Code != "context_length_exceeded" {
		t.Fatalf("error.code = %q, want context_length_exceeded", eb.Error.Code)
	}
	if eb.Error.Type != "invalid_request_error" {
		t.Fatalf("error.type = %q, want invalid_request_error", eb.Error.Type)
	}
	if !strings.Contains(eb.Error.Message, "prompt is too long") {
		t.Fatalf("error.message = %q, want it to describe the oversized prompt", eb.Error.Message)
	}
}

func TestChatPromptTooLongReturns400(t *testing.T) {
	ts, eng := newContextLengthServer(t, 1024, nil)
	long := strings.Repeat("x", 2000)
	body := fmt.Sprintf(`{"model":"alpha","messages":[{"role":"user","content":%q}]}`, long)

	resp := postJSON(t, ts.URL+"/v1/chat/completions", body)
	defer resp.Body.Close()

	assertContextLengthError(t, resp)
	if got := eng.generateCalls.Load(); got != 0 {
		t.Fatalf("Generate called %d time(s), want 0 (rejected before generation)", got)
	}
}

func TestChatPromptTooLongStreamReturns400(t *testing.T) {
	ts, eng := newContextLengthServer(t, 1024, nil)
	long := strings.Repeat("x", 2000)
	body := fmt.Sprintf(`{"model":"alpha","stream":true,"messages":[{"role":"user","content":%q}]}`, long)

	resp := postJSON(t, ts.URL+"/v1/chat/completions", body)
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "data:") {
		t.Fatalf("streaming response opened an SSE stream: %s", raw)
	}
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", resp.StatusCode, raw)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Fatalf("Content-Type = %q, want JSON", ct)
	}
	var eb errorBody
	if err := json.Unmarshal(raw, &eb); err != nil {
		t.Fatal(err)
	}
	if eb.Error.Code != "context_length_exceeded" || eb.Error.Type != "invalid_request_error" {
		t.Fatalf("unexpected error body: %+v", eb.Error)
	}
	if got := eng.generateCalls.Load(); got != 0 {
		t.Fatalf("Generate called %d time(s), want 0 (rejected before generation)", got)
	}
}

func TestChatPromptAtWindowStreams(t *testing.T) {
	ts, eng := newContextLengthServer(t, 1024, nil)
	exact := strings.Repeat("x", 1024)
	body := fmt.Sprintf(`{"model":"alpha","stream":true,"messages":[{"role":"user","content":%q}]}`, exact)

	resp := postJSON(t, ts.URL+"/v1/chat/completions", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200 (a prompt exactly at the window must pass); body=%s", resp.StatusCode, raw)
	}

	var sawDone, sawStop bool
	for _, ev := range readTestSSE(t, resp.Body) {
		if ev.Data == "[DONE]" {
			sawDone = true
			continue
		}
		var chunk chatChunk
		if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
			t.Fatalf("bad chunk %q: %v", ev.Data, err)
		}
		for _, c := range chunk.Choices {
			if c.FinishReason == "stop" {
				sawStop = true
			}
		}
	}
	if !sawDone {
		t.Fatal("stream did not terminate with [DONE]")
	}
	if !sawStop {
		t.Fatal("stream did not report finish_reason stop")
	}
	if got := eng.generateCalls.Load(); got != 1 {
		t.Fatalf("Generate called %d time(s), want 1", got)
	}
}

func TestResponsesPromptTooLongReturns400(t *testing.T) {
	ts, eng := newContextLengthServer(t, 1024, nil)
	long := strings.Repeat("x", 2000)
	body := fmt.Sprintf(`{"model":"alpha","input":%q}`, long)

	resp := postJSON(t, ts.URL+"/v1/responses", body)
	defer resp.Body.Close()

	assertContextLengthError(t, resp)
	if got := eng.generateCalls.Load(); got != 0 {
		t.Fatalf("Generate called %d time(s), want 0 (rejected before generation)", got)
	}
}

func TestMessagesPromptTooLongReturns400(t *testing.T) {
	ts, eng := newContextLengthServer(t, 1024, nil)
	long := strings.Repeat("x", 2000)
	body := fmt.Sprintf(`{"model":"alpha","max_tokens":64,"messages":[{"role":"user","content":%q}]}`, long)

	resp := postJSON(t, ts.URL+"/v1/messages", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 400; body=%s", resp.StatusCode, raw)
	}
	var eb anthropicErrorBody
	if err := json.NewDecoder(resp.Body).Decode(&eb); err != nil {
		t.Fatal(err)
	}
	if eb.Type != "error" || eb.Error.Type != "invalid_request_error" {
		t.Fatalf("unexpected Anthropic error body: %+v", eb)
	}
	if !strings.Contains(eb.Error.Message, "prompt is too long") {
		t.Fatalf("error.message = %q, want it to describe the oversized prompt", eb.Error.Message)
	}
	if got := eng.generateCalls.Load(); got != 0 {
		t.Fatalf("Generate called %d time(s), want 0 (rejected before generation)", got)
	}
}

// TestEngineContextRefusalMapped covers the fallback path: the preflight passes
// (a fitting prompt count) but the engine itself refuses the prompt mid-request.
// The non-streaming response must be the same 400 body, and a stream must carry
// the code in its SSE error frame and then terminate with [DONE].
func TestEngineContextRefusalMapped(t *testing.T) {
	refusal := fmt.Errorf("%w: prompt is too long: 2000 tokens > 1024 maximum", engine.ErrContextLength)

	t.Run("non-stream", func(t *testing.T) {
		ts, _ := newContextLengthServer(t, 1024, func(e *windowEngine) {
			e.promptTokens = 1 // preflight passes; only the engine refuses
			e.genErr = refusal
		})
		body := `{"model":"alpha","prompt":"hi","max_tokens":8}`
		resp := postJSON(t, ts.URL+"/v1/completions", body)
		defer resp.Body.Close()
		assertContextLengthError(t, resp)
	})

	t.Run("stream", func(t *testing.T) {
		ts, _ := newContextLengthServer(t, 1024, func(e *windowEngine) {
			e.promptTokens = 1 // preflight passes; only the engine refuses
			e.genErr = refusal
		})
		body := `{"model":"alpha","stream":true,"messages":[{"role":"user","content":"hi"}]}`
		resp := postJSON(t, ts.URL+"/v1/chat/completions", body)
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			raw, _ := io.ReadAll(resp.Body)
			t.Fatalf("status = %d, want 200 (headers already sent); body=%s", resp.StatusCode, raw)
		}

		var (
			sawCode bool
			sawDone bool
			doneAt  = -1
			errAt   = -1
		)
		events := readTestSSE(t, resp.Body)
		for i, ev := range events {
			if ev.Data == "[DONE]" {
				sawDone = true
				doneAt = i
				continue
			}
			if !strings.Contains(ev.Data, `"error"`) {
				continue
			}
			errAt = i
			if !strings.Contains(ev.Data, `"code":"context_length_exceeded"`) {
				t.Fatalf("SSE error frame missing the context_length_exceeded code: %s", ev.Data)
			}
			var eb errorBody
			if err := json.Unmarshal([]byte(ev.Data), &eb); err != nil {
				t.Fatalf("bad SSE error frame %q: %v", ev.Data, err)
			}
			if eb.Error.Code != "context_length_exceeded" || eb.Error.Type != "invalid_request_error" {
				t.Fatalf("unexpected SSE error payload: %+v", eb.Error)
			}
			if !strings.HasPrefix(eb.Error.Message, "generation failed: ") {
				t.Fatalf("error message = %q, want the generation failed prefix", eb.Error.Message)
			}
			sawCode = true
		}
		if !sawCode {
			t.Fatalf("no SSE error frame carrying the context code; events=%+v", events)
		}
		if !sawDone {
			t.Fatal("stream did not terminate with [DONE] after the error")
		}
		if errAt > doneAt {
			t.Fatal("SSE error frame appeared after [DONE]")
		}
	})
}

func TestManagerPromptTokensFallsBackToRegistryWindow(t *testing.T) {
	cfg := testConfig()
	cfg.MaxSeqLen = 8192
	cfg.Models = []config.RegistryModel{
		{ID: "alpha", Name: "Alpha", File: "alpha.gguf", ContextLength: 4096},
	}
	// The engine reports a zero window (as a model whose runtime metadata does
	// not expose one would); the manager must fall back to the registry entry.
	zero := 0
	mgr := NewManager(cfg, func(id string) engine.Engine {
		return &windowEngine{
			Stub:         engine.NewStub(id),
			promptTokens: 7,
			infoWindow:   &zero,
		}
	}, slog.New(slog.NewTextHandler(io.Discard, nil)))

	tokens, window, err := mgr.PromptTokens(context.Background(), "alpha", engine.GenerateRequest{Prompt: "hi"})
	if err != nil {
		t.Fatal(err)
	}
	if tokens != 7 {
		t.Fatalf("prompt tokens = %d, want 7", tokens)
	}
	if window != 4096 {
		t.Fatalf("window = %d, want the registry context_length 4096", window)
	}
}
