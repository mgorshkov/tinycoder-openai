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
	// countCalls counts CountPromptTokens invocations, which (unlike Generate)
	// the stub does not call internally. A test uses it to prove the preflight
	// actually tokenized the prompt.
	countCalls atomic.Int32
	// promptTokens, when positive, is the count returned for every request.
	// Zero makes the engine count the request's characters instead (one per
	// byte) so a test can drive the boundary precisely.
	promptTokens int
	// infoWindow, when non-nil, overrides Info().ContextLength.
	infoWindow *int
	// zeroInfo makes Info() return the zero ModelInfo (no runtime window).
	zeroInfo bool
	// genErr, when non-nil, is returned from Generate (after load).
	genErr error
}

func (e *windowEngine) Info() engine.ModelInfo {
	if e.zeroInfo {
		return engine.ModelInfo{}
	}
	info := e.Stub.Info()
	if e.infoWindow != nil {
		info.ContextLength = *e.infoWindow
	}
	return info
}

func (e *windowEngine) CountPromptTokens(req engine.GenerateRequest) (int, error) {
	e.countCalls.Add(1)
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
// context window and whose engine is the shared windowEngine returned. The
// manager is returned so a test can warm (load) the model before a streaming
// request, which is when R1 runs the preflight.
func newContextLengthServer(t *testing.T, window int, configure func(*windowEngine)) (*httptest.Server, *windowEngine, *Manager) {
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
	return ts, eng, mgr
}

// warmModel loads and releases the shared "alpha" model so the next streaming
// request is warm and therefore runs the pre-stream window preflight.
func warmModel(t *testing.T, mgr *Manager) {
	t.Helper()
	lease, err := mgr.Acquire(context.Background(), "alpha")
	if err != nil {
		t.Fatalf("warming model: %v", err)
	}
	lease.Release()
}

// assertNoSSEFrames fails when a response body contains any SSE event/data
// frame (used to prove a streaming request was rejected before the stream
// opened).
func assertNoSSEFrames(t *testing.T, raw []byte) {
	t.Helper()
	body := string(raw)
	if strings.Contains(body, "data:") || strings.Contains(body, "event:") {
		t.Fatalf("streaming response opened an SSE stream: %s", body)
	}
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
	ts, eng, _ := newContextLengthServer(t, 1024, nil)
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
	ts, eng, mgr := newContextLengthServer(t, 1024, nil)
	// R1: a streaming preflight runs only for a warm model.
	warmModel(t, mgr)
	long := strings.Repeat("x", 2000)
	body := fmt.Sprintf(`{"model":"alpha","stream":true,"messages":[{"role":"user","content":%q}]}`, long)

	resp := postJSON(t, ts.URL+"/v1/chat/completions", body)
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	assertNoSSEFrames(t, raw)
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

func TestResponsesPromptTooLongStreamReturns400(t *testing.T) {
	ts, eng, mgr := newContextLengthServer(t, 1024, nil)
	// R1: a streaming preflight runs only for a warm model.
	warmModel(t, mgr)
	long := strings.Repeat("x", 2000)
	body := fmt.Sprintf(`{"model":"alpha","stream":true,"input":%q}`, long)

	resp := postJSON(t, ts.URL+"/v1/responses", body)
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	assertNoSSEFrames(t, raw)
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

func TestMessagesPromptTooLongStreamReturns400(t *testing.T) {
	ts, eng, mgr := newContextLengthServer(t, 1024, nil)
	// R1: a streaming preflight runs only for a warm model.
	warmModel(t, mgr)
	long := strings.Repeat("x", 2000)
	body := fmt.Sprintf(`{"model":"alpha","stream":true,"max_tokens":64,"messages":[{"role":"user","content":%q}]}`, long)

	resp := postJSON(t, ts.URL+"/v1/messages", body)
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	assertNoSSEFrames(t, raw)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", resp.StatusCode, raw)
	}
	var eb anthropicErrorBody
	if err := json.Unmarshal(raw, &eb); err != nil {
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

func TestChatPromptAtWindowStreams(t *testing.T) {
	ts, eng, mgr := newContextLengthServer(t, 1024, nil)
	// Warm the model so the streaming preflight actually runs: the boundary is
	// only exercised when the request is cold otherwise (the check is skipped).
	warmModel(t, mgr)
	// The fake counts one token per byte, so this prompt is exactly the window.
	exact := strings.Repeat("x", 1024)
	body := fmt.Sprintf(`{"model":"alpha","stream":true,"messages":[{"role":"user","content":%q}]}`, exact)

	resp := postJSON(t, ts.URL+"/v1/chat/completions", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200 (a prompt exactly at the window must pass); body=%s", resp.StatusCode, raw)
	}

	var sawContent, sawDone, sawStop bool
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
			if c.Delta.Content != "" {
				sawContent = true
			}
			if c.FinishReason == "stop" {
				sawStop = true
			}
		}
	}
	if !sawContent {
		t.Fatal("stream did not carry any generated content")
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
	// Prove the warm streaming preflight actually tokenized the prompt (rather
	// than the request being cold and skipping the check): this is what makes
	// the exact-boundary assertion meaningful.
	if got := eng.countCalls.Load(); got < 1 {
		t.Fatal("the streaming preflight never counted the prompt; the exact-window boundary was not exercised")
	}
}

func TestResponsesPromptTooLongReturns400(t *testing.T) {
	ts, eng, _ := newContextLengthServer(t, 1024, nil)
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
	ts, eng, _ := newContextLengthServer(t, 1024, nil)
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

// TestEngineContextRefusalMapped covers the fallback path: the preflight does
// not reject the request (non-streaming, it sees a fitting prompt count; the
// streaming subtest runs cold, so its preflight is skipped) but the engine
// itself refuses the prompt mid-request. The non-streaming response must be the
// same 400 body, and a stream must carry the code in its SSE error frame and
// then terminate with [DONE].
func TestEngineContextRefusalMapped(t *testing.T) {
	refusal := fmt.Errorf("%w: prompt is too long: 2000 tokens > 1024 maximum", engine.ErrContextLength)

	t.Run("non-stream", func(t *testing.T) {
		ts, _, _ := newContextLengthServer(t, 1024, func(e *windowEngine) {
			e.promptTokens = 1 // preflight passes; only the engine refuses
			e.genErr = refusal
		})
		body := `{"model":"alpha","prompt":"hi","max_tokens":8}`
		resp := postJSON(t, ts.URL+"/v1/completions", body)
		defer resp.Body.Close()
		assertContextLengthError(t, resp)
	})

	t.Run("stream", func(t *testing.T) {
		ts, _, _ := newContextLengthServer(t, 1024, func(e *windowEngine) {
			e.promptTokens = 1 // a fitting count; only the engine refuses
			e.genErr = refusal
		})
		// This request is cold, so the streaming preflight is skipped entirely;
		// the "preflight passes" wording would be false here. The engine refusal
		// is what must be mapped.
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

// TestColdChatStreamEngineRefusalMapped covers R1's cold streaming path: the
// stream opens immediately (role chunk), so the preflight is skipped and the
// engine refusal is mapped into the SSE error event with the context code.
func TestColdChatStreamEngineRefusalMapped(t *testing.T) {
	refusal := fmt.Errorf("%w: prompt is too long: 2000 tokens > 1024 maximum", engine.ErrContextLength)
	ts, eng, _ := newContextLengthServer(t, 1024, func(e *windowEngine) {
		e.promptTokens = 2000 // a preflight would reject; cold streaming must skip it
		e.genErr = refusal
	})

	body := `{"model":"alpha","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	resp := postJSON(t, ts.URL+"/v1/chat/completions", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200 (headers already sent); body=%s", resp.StatusCode, raw)
	}

	events := readTestSSE(t, resp.Body)
	if len(events) < 3 {
		t.Fatalf("expected a role chunk, an error frame and [DONE]; got %+v", events)
	}
	var first chatChunk
	if err := json.Unmarshal([]byte(events[0].Data), &first); err != nil {
		t.Fatalf("bad first event %q: %v", events[0].Data, err)
	}
	if len(first.Choices) == 0 || first.Choices[0].Delta.Role != "assistant" {
		t.Fatalf("first event is not the assistant role chunk: %s", events[0].Data)
	}

	errAt, doneAt := -1, -1
	for i, ev := range events {
		if ev.Data == "[DONE]" {
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
	}
	if errAt < 0 {
		t.Fatalf("no SSE error frame; events=%+v", events)
	}
	if doneAt != len(events)-1 {
		t.Fatalf("[DONE] is not the terminal event (doneAt=%d, events=%d)", doneAt, len(events))
	}
	if errAt >= doneAt {
		t.Fatal("SSE error frame appeared after [DONE]")
	}
	if got := eng.generateCalls.Load(); got != 1 {
		t.Fatalf("Generate called %d time(s), want 1", got)
	}
}

// TestColdResponsesStreamEngineRefusalMapped covers the same cold path on the
// Responses API: the first event arrives before the failure, and the terminal
// response.failed event carries the context code.
func TestColdResponsesStreamEngineRefusalMapped(t *testing.T) {
	refusal := fmt.Errorf("%w: prompt is too long: 2000 tokens > 1024 maximum", engine.ErrContextLength)
	ts, eng, _ := newContextLengthServer(t, 1024, func(e *windowEngine) {
		e.promptTokens = 2000 // a preflight would reject; cold streaming must skip it
		e.genErr = refusal
	})

	body := `{"model":"alpha","stream":true,"input":"hi"}`
	resp := postJSON(t, ts.URL+"/v1/responses", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200 (headers already sent); body=%s", resp.StatusCode, raw)
	}

	events := readTestSSE(t, resp.Body)
	if len(events) == 0 || events[0].Name != "response.created" {
		t.Fatalf("first event is not response.created; events=%+v", events)
	}
	failedAt := -1
	for i, ev := range events {
		switch ev.Name {
		case "response.completed":
			t.Fatal("a refused generation must not emit response.completed")
		case "response.failed":
			failedAt = i
			if !strings.Contains(ev.Data, `"code":"context_length_exceeded"`) {
				t.Fatalf("response.failed missing the context code: %s", ev.Data)
			}
			var failed struct {
				Response struct {
					Error struct {
						Code    string `json:"code"`
						Message string `json:"message"`
					} `json:"error"`
				} `json:"response"`
			}
			if err := json.Unmarshal([]byte(ev.Data), &failed); err != nil {
				t.Fatalf("bad response.failed frame %q: %v", ev.Data, err)
			}
			if failed.Response.Error.Code != "context_length_exceeded" {
				t.Fatalf("response.failed code = %q, want context_length_exceeded", failed.Response.Error.Code)
			}
			if !strings.HasPrefix(failed.Response.Error.Message, "generation failed: ") {
				t.Fatalf("response.failed message = %q", failed.Response.Error.Message)
			}
		}
	}
	if failedAt != len(events)-1 {
		t.Fatalf("response.failed is not the terminal event (failedAt=%d, events=%d)", failedAt, len(events))
	}
	if got := eng.generateCalls.Load(); got != 1 {
		t.Fatalf("Generate called %d time(s), want 1", got)
	}
}

// TestColdMessagesStreamEngineRefusalMapped covers the cold path on the
// Anthropic Messages API: message_start arrives first, then an error event of
// type invalid_request_error (Anthropic errors carry no code), and no
// message_stop follows.
func TestColdMessagesStreamEngineRefusalMapped(t *testing.T) {
	refusal := fmt.Errorf("%w: prompt is too long: 2000 tokens > 1024 maximum", engine.ErrContextLength)
	ts, eng, _ := newContextLengthServer(t, 1024, func(e *windowEngine) {
		e.promptTokens = 2000 // a preflight would reject; cold streaming must skip it
		e.genErr = refusal
	})

	body := `{"model":"alpha","stream":true,"max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`
	resp := postJSON(t, ts.URL+"/v1/messages", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200 (headers already sent); body=%s", resp.StatusCode, raw)
	}

	events := readTestSSE(t, resp.Body)
	if len(events) == 0 || events[0].Name != "message_start" {
		t.Fatalf("first event is not message_start; events=%+v", events)
	}
	errorAt := -1
	for i, ev := range events {
		if ev.Name == "message_stop" {
			t.Fatal("a refused generation must not emit message_stop")
		}
		if ev.Name != "error" {
			continue
		}
		errorAt = i
		var eb anthropicErrorBody
		if err := json.Unmarshal([]byte(ev.Data), &eb); err != nil {
			t.Fatalf("bad error event %q: %v", ev.Data, err)
		}
		if eb.Type != "error" || eb.Error.Type != "invalid_request_error" {
			t.Fatalf("unexpected Anthropic error event: %+v", eb)
		}
		if !strings.HasPrefix(eb.Error.Message, "generation failed: ") {
			t.Fatalf("error message = %q, want the generation failed prefix", eb.Error.Message)
		}
	}
	if errorAt != len(events)-1 {
		t.Fatalf("error event is not terminal (errorAt=%d, events=%d)", errorAt, len(events))
	}
	if got := eng.generateCalls.Load(); got != 1 {
		t.Fatalf("Generate called %d time(s), want 1", got)
	}
}

// TestMessagesEngineContextRefusalMapped covers the engine-refusal fallback on
// the non-streaming and (warm) streaming Messages endpoints.
func TestMessagesEngineContextRefusalMapped(t *testing.T) {
	refusal := fmt.Errorf("%w: prompt is too long: 2000 tokens > 1024 maximum", engine.ErrContextLength)

	t.Run("non-stream", func(t *testing.T) {
		ts, _, _ := newContextLengthServer(t, 1024, func(e *windowEngine) {
			e.promptTokens = 1 // preflight passes; only the engine refuses
			e.genErr = refusal
		})
		body := `{"model":"alpha","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`
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
	})

	t.Run("stream", func(t *testing.T) {
		ts, _, mgr := newContextLengthServer(t, 1024, func(e *windowEngine) {
			e.promptTokens = 1 // preflight passes; only the engine refuses
			e.genErr = refusal
		})
		warmModel(t, mgr)
		body := `{"model":"alpha","stream":true,"max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`
		resp := postJSON(t, ts.URL+"/v1/messages", body)
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			raw, _ := io.ReadAll(resp.Body)
			t.Fatalf("status = %d, want 200 (headers already sent); body=%s", resp.StatusCode, raw)
		}
		events := readTestSSE(t, resp.Body)
		found := false
		for _, ev := range events {
			if ev.Name != "error" {
				continue
			}
			found = true
			var eb anthropicErrorBody
			if err := json.Unmarshal([]byte(ev.Data), &eb); err != nil {
				t.Fatalf("bad error event %q: %v", ev.Data, err)
			}
			if eb.Error.Type != "invalid_request_error" {
				t.Fatalf("error type = %q, want invalid_request_error", eb.Error.Type)
			}
		}
		if !found {
			t.Fatalf("no error event; events=%+v", events)
		}
	})
}

// TestResponsesEngineContextRefusalMapped covers the engine-refusal fallback on
// the non-streaming and (warm) streaming Responses endpoints.
func TestResponsesEngineContextRefusalMapped(t *testing.T) {
	refusal := fmt.Errorf("%w: prompt is too long: 2000 tokens > 1024 maximum", engine.ErrContextLength)

	t.Run("non-stream", func(t *testing.T) {
		ts, _, _ := newContextLengthServer(t, 1024, func(e *windowEngine) {
			e.promptTokens = 1 // preflight passes; only the engine refuses
			e.genErr = refusal
		})
		body := `{"model":"alpha","input":"hi"}`
		resp := postJSON(t, ts.URL+"/v1/responses", body)
		defer resp.Body.Close()
		assertContextLengthError(t, resp)
	})

	t.Run("stream", func(t *testing.T) {
		ts, _, mgr := newContextLengthServer(t, 1024, func(e *windowEngine) {
			e.promptTokens = 1 // preflight passes; only the engine refuses
			e.genErr = refusal
		})
		warmModel(t, mgr)
		body := `{"model":"alpha","stream":true,"input":"hi"}`
		resp := postJSON(t, ts.URL+"/v1/responses", body)
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			raw, _ := io.ReadAll(resp.Body)
			t.Fatalf("status = %d, want 200 (headers already sent); body=%s", resp.StatusCode, raw)
		}
		found := false
		for _, ev := range readTestSSE(t, resp.Body) {
			if ev.Name != "response.failed" {
				continue
			}
			found = true
			if !strings.Contains(ev.Data, `"code":"context_length_exceeded"`) {
				t.Fatalf("response.failed missing the context code: %s", ev.Data)
			}
		}
		if !found {
			t.Fatal("no response.failed event")
		}
	})
}

func TestManagerPromptTokensFallsBackToRegistryWindow(t *testing.T) {
	cfg := testConfig()
	cfg.MaxSeqLen = 8192
	cfg.Models = []config.RegistryModel{
		{ID: "alpha", Name: "Alpha", File: "alpha.gguf", ContextLength: 4096},
	}
	// The engine reports the zero ModelInfo (no runtime window); the manager
	// must fall back to the registry entry.
	mgr := NewManager(cfg, func(id string) engine.Engine {
		return &windowEngine{
			Stub:         engine.NewStub(id),
			promptTokens: 7,
			zeroInfo:     true,
		}
	}, slog.New(slog.NewTextHandler(io.Discard, nil)))

	// Load the model first: Acquire backfills a zero runtime ContextLength from
	// opts.MaxSeqLen, which would otherwise mask the fallback under test. Clear
	// the cached info after warming so PromptTokens sees the real zero-window
	// engine metadata and must consult the registry.
	warmModel(t, mgr)
	mgr.mu.Lock()
	e := mgr.entries["alpha"]
	mgr.mu.Unlock()
	e.mu.Lock()
	e.info = engine.ModelInfo{}
	e.mu.Unlock()

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
