package server

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mgorshkov/tinycoder-openai/internal/config"
	"github.com/mgorshkov/tinycoder-openai/internal/engine"
)

func testConfig() config.Config {
	cfg := config.Default()
	cfg.MaxConcurrent = 2
	cfg.Models = []config.RegistryModel{
		{ID: "alpha", Name: "Alpha", File: "alpha.gguf", ContextLength: 4096},
		{ID: "beta", Name: "Beta", File: "beta.gguf", ContextLength: 2048},
	}
	return cfg
}

func newTestServer(t *testing.T, cfg config.Config) *httptest.Server {
	t.Helper()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	mgr := NewManager(cfg, func(id string) engine.Engine { return engine.NewStub(id) }, log)
	srv := New(cfg, mgr, log)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts
}

// decodeJSON decodes a JSON body (helper shared by the timeout tests).
func decodeJSON(r io.Reader, v any) error { return json.NewDecoder(r).Decode(v) }

func TestListModels(t *testing.T) {
	ts := newTestServer(t, testConfig())
	resp, err := http.Get(ts.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	var body modelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body.Object != "list" || len(body.Data) != 2 {
		t.Fatalf("unexpected body: %+v", body)
	}
	if body.Data[0].ID != "alpha" || body.Data[0].ContextLength != 4096 {
		t.Fatalf("unexpected first model: %+v", body.Data[0])
	}
}

func TestCanonicalV1Routes(t *testing.T) {
	ts := newTestServer(t, testConfig())
	for _, path := range []string{"/v1/models"} {
		resp, err := http.Get(ts.URL + path)
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("%s: status = %d", path, resp.StatusCode)
		}
	}
}

func TestGenerateNonStream(t *testing.T) {
	ts := newTestServer(t, testConfig())
	body := `{"model":"alpha","prompt":"hello world","max_tokens":16}`
	resp, err := http.Post(ts.URL+"/v1/generate", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d body=%s", resp.StatusCode, b)
	}
	var out completionResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Model != "alpha" || len(out.Choices) != 1 {
		t.Fatalf("unexpected: %+v", out)
	}
	if !strings.Contains(out.Choices[0].Text, "stub:alpha") {
		t.Fatalf("unexpected text: %q", out.Choices[0].Text)
	}
	if out.Usage.PromptTokens < 1 || out.Usage.CompletionTokens < 1 {
		t.Fatalf("unexpected usage: %+v", out.Usage)
	}
}

func TestChatCompletionsNonStream(t *testing.T) {
	ts := newTestServer(t, testConfig())
	body := `{"model":"beta","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d body=%s", resp.StatusCode, b)
	}
	var out chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Object != "chat.completion" || out.Choices[0].Message.Role != "assistant" {
		t.Fatalf("unexpected: %+v", out)
	}
}

func TestChatCompletionsArrayContent(t *testing.T) {
	ts := newTestServer(t, testConfig())
	body := `{"model":"beta","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"text","text":"there"}]}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d body=%s", resp.StatusCode, b)
	}
	var out chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if len(out.Choices) != 1 || out.Choices[0].Message.Role != "assistant" {
		t.Fatalf("unexpected: %+v", out)
	}
}

func TestChatCompletionsArrayContentStreaming(t *testing.T) {
	ts := newTestServer(t, testConfig())
	body := `{"model":"alpha","messages":[{"role":"user","content":[{"type":"text","text":"stream"}]}],"stream":true,"max_tokens":8}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d body=%s", resp.StatusCode, b)
	}
}

func TestMessageContentUnmarshal(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"string", `"hello"`, "hello"},
		{"null", `null`, ""},
		{"array", `[{"type":"text","text":"a"},{"type":"text","text":"b"}]`, "a\nb"},
		{"array with non-text", `[{"type":"image_url","image_url":{"url":"x"}},{"type":"text","text":"b"}]`, "b"},
		{"empty array", `[]`, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var m chatMessage
			if err := json.Unmarshal([]byte(`{"role":"user","content":`+tc.in+`}`), &m); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if m.Content.text != tc.want {
				t.Fatalf("got %q, want %q", m.Content.text, tc.want)
			}
		})
	}
}

func TestMessageContentUnmarshalRejectsScalar(t *testing.T) {
	var m chatMessage
	err := json.Unmarshal([]byte(`{"role":"user","content":42}`), &m)
	if err == nil {
		t.Fatal("expected error for numeric content")
	}
}

func TestChatStreaming(t *testing.T) {
	ts := newTestServer(t, testConfig())
	body := `{"model":"alpha","messages":[{"role":"user","content":"stream please"}],"stream":true,"max_tokens":8}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type = %q", ct)
	}

	sc := bufio.NewScanner(resp.Body)
	var (
		chunks int
		text   strings.Builder
		done   bool
	)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			done = true
			break
		}
		chunks++
		var chunk chatChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("bad chunk %q: %v", payload, err)
		}
		if len(chunk.Choices) > 0 {
			text.WriteString(chunk.Choices[0].Delta.Content)
		}
	}
	if !done {
		t.Fatal("stream did not terminate with [DONE]")
	}
	if chunks < 2 {
		t.Fatalf("expected multiple chunks, got %d", chunks)
	}
	if !strings.Contains(text.String(), "stub:alpha") {
		t.Fatalf("streamed text = %q", text.String())
	}
}

// TestStreamOptionsAccepted guards against the regression where a standard
// OpenAI stream_options field caused "unknown field" 400 errors.
func TestStreamOptionsAccepted(t *testing.T) {
	ts := newTestServer(t, testConfig())
	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":8,"stream_options":{"include_usage":true}}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d body=%s", resp.StatusCode, b)
	}
}

// TestUnknownFieldsIgnored verifies that optional OpenAI fields the server does
// not model do not cause the request to be rejected.
func TestUnknownFieldsIgnored(t *testing.T) {
	ts := newTestServer(t, testConfig())
	body := `{"model":"alpha","prompt":"hi","max_tokens":8,"response_format":{"type":"text"},"frequency_penalty":0.5,"logprobs":true,"tool_choice":"auto"}`
	resp, err := http.Post(ts.URL+"/v1/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d body=%s", resp.StatusCode, b)
	}
}

// TestStreamIncludeUsage verifies that stream_options.include_usage produces a
// final usage-bearing chunk before [DONE].
func TestStreamIncludeUsage(t *testing.T) {
	ts := newTestServer(t, testConfig())
	body := `{"model":"alpha","messages":[{"role":"user","content":"usage please"}],"stream":true,"max_tokens":8,"stream_options":{"include_usage":true}}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d body=%s", resp.StatusCode, b)
	}

	sc := bufio.NewScanner(resp.Body)
	var (
		sawUsage bool
		done     bool
	)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			done = true
			break
		}
		var chunk chatChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("bad chunk %q: %v", payload, err)
		}
		if chunk.Usage != nil {
			sawUsage = true
			if chunk.Usage.TotalTokens < 1 {
				t.Fatalf("unexpected usage: %+v", chunk.Usage)
			}
		}
	}
	if !done {
		t.Fatal("stream did not terminate with [DONE]")
	}
	if !sawUsage {
		t.Fatal("expected a usage chunk when include_usage is true")
	}
}

func TestUnknownModel404(t *testing.T) {
	ts := newTestServer(t, testConfig())
	body := `{"model":"nope","prompt":"x"}`
	resp, err := http.Post(ts.URL+"/v1/generate", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	var eb errorBody
	if err := json.NewDecoder(resp.Body).Decode(&eb); err != nil {
		t.Fatal(err)
	}
	if eb.Error.Code != "model_not_found" {
		t.Fatalf("unexpected error: %+v", eb)
	}
}

// TestModelNotInConfigIsRejected verifies that only models declared in the
// config are served. Even when the underlying manager has an entry for a model
// (e.g. one discovered on disk), a request for it must 404 on every generation
// endpoint.
func TestModelNotInConfigIsRejected(t *testing.T) {
	cfg := testConfig()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))

	// The manager is built from a superset registry that includes "ghost", but
	// the Server is constructed with cfg, which does not declare it.
	extra := config.Default()
	extra.Models = append(append([]config.RegistryModel{}, cfg.Models...),
		config.RegistryModel{ID: "ghost", Name: "Ghost", File: "ghost.gguf"})
	mgr := NewManager(extra, func(id string) engine.Engine { return engine.NewStub(id) }, log)
	ts := httptest.NewServer(New(cfg, mgr, log).Handler())
	t.Cleanup(ts.Close)

	// The ghost model must not be listed.
	resp, err := http.Get(ts.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var list modelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		t.Fatal(err)
	}
	for _, m := range list.Data {
		if m.ID == "ghost" {
			t.Fatalf("/v1/models must not advertise a model absent from config")
		}
	}

	// Every generation endpoint must reject it with 404 model_not_found.
	for _, path := range []string{"/v1/generate", "/v1/completions", "/v1/chat/completions"} {
		body := `{"model":"ghost","prompt":"hi","messages":[{"role":"user","content":"hi"}]}`
		resp, err := http.Post(ts.URL+path, "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		if resp.StatusCode != http.StatusNotFound {
			resp.Body.Close()
			t.Fatalf("%s: status = %d, want 404", path, resp.StatusCode)
		}
		var eb errorBody
		if err := json.NewDecoder(resp.Body).Decode(&eb); err != nil {
			resp.Body.Close()
			t.Fatalf("%s: %v", path, err)
		}
		resp.Body.Close()
		if eb.Error.Code != "model_not_found" {
			t.Fatalf("%s: unexpected error: %+v", path, eb)
		}
	}
}

func TestMissingPrompt400(t *testing.T) {
	ts := newTestServer(t, testConfig())
	resp, err := http.Post(ts.URL+"/v1/generate", "application/json", strings.NewReader(`{"model":"alpha"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d", resp.StatusCode)
	}
}

func TestAuthEnforced(t *testing.T) {
	cfg := testConfig()
	cfg.APIKeys = []string{"secret"}
	ts := newTestServer(t, cfg)

	resp, err := http.Get(ts.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", resp.StatusCode)
	}

	req, _ := http.NewRequest(http.MethodGet, ts.URL+"/v1/models", nil)
	req.Header.Set("Authorization", "Bearer secret")
	resp2, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 with token, got %d", resp2.StatusCode)
	}
}

// TestAuthMultipleKeys verifies that every key from api_keys is accepted, while
// an unknown key is rejected with 401.
func TestAuthMultipleKeys(t *testing.T) {
	cfg := testConfig()
	cfg.APIKeys = []string{"sk-one", "sk-two", ""}
	ts := newTestServer(t, cfg)

	call := func(key string) int {
		req, _ := http.NewRequest(http.MethodGet, ts.URL+"/v1/models", nil)
		if key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}

	for _, key := range []string{"sk-one", "sk-two"} {
		if got := call(key); got != http.StatusOK {
			t.Fatalf("key %q: status = %d, want 200", key, got)
		}
	}
	for _, key := range []string{"", "wrong", "sk-three"} {
		if got := call(key); got != http.StatusUnauthorized {
			t.Fatalf("key %q: status = %d, want 401", key, got)
		}
	}

	// The api_key query parameter is accepted as an alternative.
	resp, err := http.Get(ts.URL + "/v1/models?api_key=sk-one")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("api_key query: status = %d, want 200", resp.StatusCode)
	}
}

// TestAuthHeaderVariants verifies that the API key is accepted from every header
// form OpenAI-compatible clients use (Authorization Bearer, a bare
// Authorization value, `api-key`, `x-api-key`) and the api_key query parameter,
// while a wrong key is still rejected.
func TestAuthHeaderVariants(t *testing.T) {
	cfg := testConfig()
	cfg.APIKeys = []string{"secret"}
	ts := newTestServer(t, cfg)

	cases := []struct {
		name    string
		headers map[string]string
		query   string
		want    int
	}{
		{"bearer", map[string]string{"Authorization": "Bearer secret"}, "", http.StatusOK},
		{"bearer lowercase scheme", map[string]string{"Authorization": "bearer secret"}, "", http.StatusOK},
		{"bare authorization", map[string]string{"Authorization": "secret"}, "", http.StatusOK},
		{"api-key header", map[string]string{"api-key": "secret"}, "", http.StatusOK},
		{"x-api-key header", map[string]string{"x-api-key": "secret"}, "", http.StatusOK},
		{"X-Api-Key header", map[string]string{"X-Api-Key": "secret"}, "", http.StatusOK},
		{"query param", nil, "?api_key=secret", http.StatusOK},
		{"wrong bearer", map[string]string{"Authorization": "Bearer nope"}, "", http.StatusUnauthorized},
		{"wrong api-key", map[string]string{"api-key": "nope"}, "", http.StatusUnauthorized},
		{"missing", nil, "", http.StatusUnauthorized},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodGet, ts.URL+"/v1/models"+tc.query, nil)
			for k, v := range tc.headers {
				req.Header.Set(k, v)
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			resp.Body.Close()
			if resp.StatusCode != tc.want {
				t.Fatalf("status = %d, want %d", resp.StatusCode, tc.want)
			}
		})
	}
}

// TestStreamKeepAlive verifies that an otherwise idle SSE stream is kept alive
// with comment lines so clients do not time out while the engine loads a model
// or prefills a large prompt (during which no tokens are produced).
func TestStreamKeepAlive(t *testing.T) {
	cfg := testConfig()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))

	// Shorten the heartbeat interval so the assertion is fast.
	oldHB := sseHeartbeat
	sseHeartbeat = 50 * time.Millisecond
	t.Cleanup(func() { sseHeartbeat = oldHB })

	release := make(chan struct{})
	mgr := NewManager(cfg, func(id string) engine.Engine {
		return &blockingEngine{Stub: engine.NewStub(id), release: release}
	}, log)
	ts := httptest.NewServer(New(cfg, mgr, log).Handler())
	t.Cleanup(ts.Close)
	t.Cleanup(func() { close(release) })

	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":4}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	lineCh := make(chan string, 16)
	go func() {
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			lineCh <- sc.Text()
		}
		close(lineCh)
	}()

	deadline := time.After(5 * time.Second)
	sawComment := false
	for !sawComment {
		select {
		case line, ok := <-lineCh:
			if !ok {
				t.Fatal("stream closed before emitting a keep-alive comment")
			}
			if strings.HasPrefix(line, ": ") {
				sawComment = true
			}
		case <-deadline:
			t.Fatal("no keep-alive comment observed while the generator was blocked")
		}
	}

	// The blocked engine is released by t.Cleanup so the server shuts down
	// cleanly even if the assertion above fails early.
}

// blockingEngine blocks in Generate until release is closed, simulating the
// lengthy model-load/prefill window during which no tokens are emitted.
type blockingEngine struct {
	*engine.Stub
	release chan struct{}
}

func (e *blockingEngine) Generate(ctx context.Context, req engine.GenerateRequest, onToken engine.TokenFunc) (engine.Usage, error) {
	if !e.Loaded() {
		return engine.Usage{}, engine.ErrNotLoaded
	}
	select {
	case <-e.release:
	case <-ctx.Done():
		return engine.Usage{}, engine.ErrContextCanceled
	}
	return e.Stub.Generate(ctx, req, onToken)
}

// TestStreamErrorEvent verifies that a generation failure on a streaming request
// is reported to the client (an SSE error event) and the stream is terminated
// with [DONE] rather than hanging or being truncated silently.
func TestStreamErrorEvent(t *testing.T) {
	cfg := testConfig()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	mgr := NewManager(cfg, func(id string) engine.Engine { return &failEngine{Stub: engine.NewStub(id)} }, log)
	ts := httptest.NewServer(New(cfg, mgr, log).Handler())
	t.Cleanup(ts.Close)

	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":4}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	sc := bufio.NewScanner(resp.Body)
	var (
		sawError bool
		sawDone  bool
	)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			sawDone = true
			break
		}
		if strings.Contains(payload, `"error"`) {
			sawError = true
		}
	}
	if !sawError {
		t.Fatal("expected an SSE error event for a failed generation")
	}
	if !sawDone {
		t.Fatal("stream did not terminate with [DONE] after the error")
	}
}

// failEngine fails generation after loading, simulating a runtime generation
// error (e.g. context overflow in the engine).
type failEngine struct{ *engine.Stub }

func (e *failEngine) Generate(ctx context.Context, req engine.GenerateRequest, onToken engine.TokenFunc) (engine.Usage, error) {
	if !e.Loaded() {
		return engine.Usage{}, engine.ErrNotLoaded
	}
	return engine.Usage{}, errors.New("boom: generation failed")
}

// TestAuthDisabledWhenNoKeys verifies the server is open when no keys are set.
func TestAuthDisabledWhenNoKeys(t *testing.T) {
	cfg := testConfig()
	cfg.APIKeys = []string{"", "  "} // blank entries must not count as keys
	ts := newTestServer(t, cfg)

	resp, err := http.Get(ts.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (open server)", resp.StatusCode)
	}
}

func TestSpecServed(t *testing.T) {
	ts := newTestServer(t, testConfig())
	resp, err := http.Get(ts.URL + "/openai.json")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	data, _ := io.ReadAll(resp.Body)
	if !json.Valid(data) {
		t.Fatal("openapi.json is not valid JSON")
	}
	if !strings.Contains(string(data), "3.1.0") {
		t.Fatal("spec missing openapi version")
	}
}

// countingEngine wraps the stub to count loads for capacity tests.
type countingEngine struct {
	*engine.Stub
	loads *int32
}

func (c *countingEngine) Load(ctx context.Context, opts engine.LoadOptions) error {
	atomic.AddInt32(c.loads, 1)
	return c.Stub.Load(ctx, opts)
}

func TestManagerCapacityEviction(t *testing.T) {
	cfg := testConfig()
	cfg.MaxLoadedModels = 1
	cfg.IdleTimeoutSec = 3600 // disable idle eviction for this test
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	var loads int32
	mgr := NewManager(cfg, func(id string) engine.Engine {
		return &countingEngine{Stub: engine.NewStub(id), loads: &loads}
	}, log)

	ctx := context.Background()
	// Use alpha then beta; capacity is 1 so alpha must be evicted (idle).
	leaseA, err := mgr.Acquire(ctx, "alpha")
	if err != nil {
		t.Fatal(err)
	}
	if !mgr.IsLoaded("alpha") {
		t.Fatal("alpha should be loaded")
	}
	leaseA.Release()

	leaseB, err := mgr.Acquire(ctx, "beta")
	if err != nil {
		t.Fatal(err)
	}
	leaseB.Release()

	if mgr.IsLoaded("alpha") {
		t.Fatal("alpha should have been evicted")
	}
	if !mgr.IsLoaded("beta") {
		t.Fatal("beta should be loaded")
	}
	if got := atomic.LoadInt32(&loads); got != 2 {
		t.Fatalf("expected 2 loads, got %d", got)
	}
}

func TestManagerSerializesGenerations(t *testing.T) {
	cfg := testConfig()
	cfg.MaxConcurrent = 2
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	mgr := NewManager(cfg, func(id string) engine.Engine { return engine.NewStub(id) }, log)

	var (
		mu      sync.Mutex
		active  int
		maxSeen int
	)
	gen := func() {
		_, err := mgr.Generate(context.Background(), "alpha", engine.GenerateRequest{
			Prompt: "hi", MaxTokens: 4,
		}, func(engine.Token) bool {
			mu.Lock()
			active++
			if active > maxSeen {
				maxSeen = active
			}
			mu.Unlock()
			time.Sleep(2 * time.Millisecond)
			mu.Lock()
			active--
			mu.Unlock()
			return true
		})
		if err != nil {
			t.Errorf("generate: %v", err)
		}
	}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); gen() }()
	}
	wg.Wait()
	if maxSeen > 1 {
		t.Fatalf("expected serialized generation per model, max concurrent = %d", maxSeen)
	}
}
