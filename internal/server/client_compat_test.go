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
	"testing"
	"time"

	"github.com/mgorshkov/tinycoder-openai/internal/engine"
)

// This file pins the wire-compatibility contract with Roo Code's OpenAI
// transports. Roo Code drives the API through
//
//   - @ai-sdk/openai-compatible (createOpenAICompatible + streamText), and
//   - the OpenAI SDK (base-openai-compatible-provider).
//
// Both emit an OpenAI Chat Completions request, so the bodies below are
// reproductions of what actually reaches this server. They guard against a
// future DTO/validation change silently rejecting a real client.

// sseOutcome is the decoded result of a streaming chat request.
type sseOutcome struct {
	status  int
	ctype   string
	done    bool
	chunks  int
	content strings.Builder
}

// postChatSSE sends body to /v1/chat/completions and drains the SSE stream,
// asserting a 200 and a text/event-stream response.
func postChatSSE(t *testing.T, baseURL, body string) sseOutcome {
	t.Helper()
	resp, err := http.Post(baseURL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	out := sseOutcome{status: resp.StatusCode, ctype: resp.Header.Get("Content-Type")}
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d body=%s", resp.StatusCode, b)
	}
	if !strings.HasPrefix(out.ctype, "text/event-stream") {
		t.Fatalf("content-type = %q, want text/event-stream", out.ctype)
	}

	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue // comments / blank framing lines
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			out.done = true
			break
		}
		out.chunks++
		var chunk chatChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("bad chunk %q: %v", payload, err)
		}
		if len(chunk.Choices) > 0 {
			out.content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}
	return out
}

// TestRooCodeFirstTurnAccepted reproduces the first request Roo Code sends on a
// fresh session: a system + user message, an OpenAI tools array, tool_choice
// "auto", parallel_tool_calls, temperature 0, and stream_options.include_usage.
func TestRooCodeFirstTurnAccepted(t *testing.T) {
	ts := newTestServer(t, testConfig())

	body := `{
		"model": "alpha",
		"messages": [
			{"role": "system", "content": "You are Roo"},
			{"role": "user", "content": "test"}
		],
		"tools": [{
			"type": "function",
			"function": {
				"name": "read_file",
				"description": "Read a file",
				"parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
				"strict": false
			}
		}],
		"tool_choice": "auto",
		"parallel_tool_calls": true,
		"temperature": 0,
		"stream": true,
		"stream_options": {"include_usage": true}
	}`

	out := postChatSSE(t, ts.URL, body)
	if !out.done {
		t.Fatal("stream did not terminate with [DONE]")
	}
	if out.chunks < 2 {
		t.Fatalf("expected multiple chunks, got %d", out.chunks)
	}
	if !strings.Contains(out.content.String(), "stub:alpha") {
		t.Fatalf("unexpected streamed content: %q", out.content.String())
	}
}

// TestRooCodeToolRoundTripAccepted reproduces a second-turn request: an assistant
// message with content:null plus tool_calls, followed by a tool-role result
// message carrying tool_call_id.
func TestRooCodeToolRoundTripAccepted(t *testing.T) {
	ts := newTestServer(t, testConfig())

	body := `{
		"model": "alpha",
		"messages": [
			{"role": "system", "content": "You are Roo"},
			{"role": "user", "content": "test"},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"a.txt\"}"}}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": "file contents"}
		],
		"tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
		"tool_choice": "auto",
		"stream": true
	}`

	out := postChatSSE(t, ts.URL, body)
	if !out.done {
		t.Fatal("stream did not terminate with [DONE]")
	}
}

// TestRooCodeUserTextPartsAccepted verifies array-form user content (the AI SDK
// emits [{type:"text",text:"..."}] when a user message has other parts too).
func TestRooCodeUserTextPartsAccepted(t *testing.T) {
	ts := newTestServer(t, testConfig())

	body := `{"model":"alpha","messages":[{"role":"user","content":[{"type":"text","text":"test"}]}],"stream":true}`
	out := postChatSSE(t, ts.URL, body)
	if !out.done {
		t.Fatal("stream did not terminate with [DONE]")
	}
}

// TestUnknownReasoningFieldsIgnored verifies that provider-extension fields Roo
// Code may attach (thinking, reasoning_effort) do not cause a 400; the endpoint
// must remain wire-compatible with clients that send fields the server does not
// model.
func TestUnknownReasoningFieldsIgnored(t *testing.T) {
	ts := newTestServer(t, testConfig())

	body := `{"model":"alpha","messages":[{"role":"user","content":"test"}],"stream":true,"max_tokens":2048,"thinking":{"type":"enabled"},"reasoning_effort":"high"}`
	out := postChatSSE(t, ts.URL, body)
	if !out.done {
		t.Fatal("stream did not terminate with [DONE]")
	}
}

func intPtr(v int) *int { return &v }

// TestMaxTokensClampedToCeiling pins the boundary behavior of max_tokens against
// max_prompt_tokens (default 32768). Roo Code clamps max_tokens to 20% of the
// model's declared context window, so a model registered with a window above
// ~163k produces a value above the ceiling. The server clamps such a request to
// the ceiling instead of rejecting it, so a standard OpenAI request never fails
// with a 400.
func TestMaxTokensClampedToCeiling(t *testing.T) {
	ts := newTestServer(t, testConfig())

	cases := []struct {
		name string
		max  int
	}{
		{"small", 256},
		{"roocode 20 percent of 128k window", 25600},
		{"exact ceiling", 32768},
		{"one over ceiling", 32769},
		{"roocode 20 percent of 200k window", 40000},
		{"large", 65536},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := fmt.Sprintf(
				`{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":%d}`,
				tc.max)

			resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				b, _ := io.ReadAll(resp.Body)
				t.Fatalf("max_tokens=%d: status = %d, want 200; body=%s", tc.max, resp.StatusCode, b)
			}
		})
	}
}

// TestBuildRequestClampsMaxTokens asserts directly on the resolved engine
// request: a max_tokens above the MaxPromptTokens ceiling is silently lowered to
// it, while an in-range value is preserved.
func TestBuildRequestClampsMaxTokens(t *testing.T) {
	cfg := testConfig()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	s := New(cfg, NewManager(cfg, func(id string) engine.Engine { return engine.NewStub(id) }, log), log)

	base := func() generationRequest {
		return generationRequest{
			Model: "alpha",
			Chat:  []chatMessage{{Role: "user", Content: messageContent{text: "hi"}}},
		}
	}

	over := base()
	over.MaxTokens = intPtr(65536)
	got, _, err := s.buildRequest(over, true, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.MaxTokens != cfg.MaxPromptTokens {
		t.Fatalf("max_tokens = %d, want clamped to %d", got.MaxTokens, cfg.MaxPromptTokens)
	}

	within := base()
	within.MaxTokens = intPtr(100)
	got, _, err = s.buildRequest(within, true, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.MaxTokens != 100 {
		t.Fatalf("max_tokens = %d, want 100", got.MaxTokens)
	}
}

// TestMaxTokensBelowOneRejected verifies that an explicit non-positive
// max_tokens is still a client error (only values above the ceiling are clamped;
// there is no sensible lower clamp), and that it is typed as an invalid request.
func TestMaxTokensBelowOneRejected(t *testing.T) {
	ts := newTestServer(t, testConfig())
	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"max_tokens":0}`

	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 400; body=%s", resp.StatusCode, b)
	}
	var eb errorBody
	if err := decodeJSON(resp.Body, &eb); err != nil {
		t.Fatalf("decode error body: %v", err)
	}
	if eb.Error.Type != "invalid_request_error" {
		t.Fatalf("error type = %q, want invalid_request_error", eb.Error.Type)
	}
}

// TestMaxTokensCeilingAppliesToCompletions verifies the same clamp is applied on
// the text-completion endpoint (shared buildRequest path).
func TestMaxTokensCeilingAppliesToCompletions(t *testing.T) {
	ts := newTestServer(t, testConfig())
	body := `{"model":"alpha","prompt":"hi","max_tokens":65536}`

	resp, err := http.Post(ts.URL+"/v1/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body=%s", resp.StatusCode, b)
	}
}

// slowLoadEngine blocks in Load until released, simulating a cold, large GGUF
// load on a shared GPU that takes a long time before the first token.
type slowLoadEngine struct {
	*engine.Stub
	release chan struct{}
}

func (e *slowLoadEngine) Load(ctx context.Context, opts engine.LoadOptions) error {
	select {
	case <-e.release:
	case <-ctx.Done():
		return ctx.Err()
	}
	return e.Stub.Load(ctx, opts)
}

// TestChatStreamsRoleChunkBeforeModelLoad verifies that a streaming chat request
// emits the opening assistant-role chunk immediately, before the (potentially
// very slow) model load completes. A client that waits for the first SSE *data*
// event therefore gets an acknowledgement instead of pure silence / keep-alive
// comments while the model loads.
func TestChatStreamsRoleChunkBeforeModelLoad(t *testing.T) {
	cfg := testConfig()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))

	release := make(chan struct{})
	mgr := NewManager(cfg, func(id string) engine.Engine {
		return &slowLoadEngine{Stub: engine.NewStub(id), release: release}
	}, log)
	ts := httptest.NewServer(New(cfg, mgr, log).Handler())
	t.Cleanup(ts.Close)
	// Released after ts.Close would run; LIFO cleanup order opens it first so
	// the blocked handler can unwind cleanly.
	t.Cleanup(func() { close(release) })

	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":4}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	lineCh := make(chan string, 8)
	go func() {
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			lineCh <- sc.Text()
		}
		close(lineCh)
	}()

	deadline := time.After(3 * time.Second)
	for {
		select {
		case line, ok := <-lineCh:
			if !ok {
				t.Fatal("stream closed before emitting the role chunk")
			}
			if !strings.HasPrefix(line, "data: ") {
				continue // ignore keep-alive comments
			}
			payload := strings.TrimPrefix(line, "data: ")
			var chunk chatChunk
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				t.Fatalf("bad chunk %q: %v", payload, err)
			}
			if len(chunk.Choices) == 0 || chunk.Choices[0].Delta.Role != "assistant" {
				t.Fatalf("first data chunk is not the assistant role chunk: %s", payload)
			}
			return // success: role chunk arrived while Load is still blocked
		case <-deadline:
			t.Fatal("no data chunk before model load completed (client sees only silence)")
		}
	}
}
