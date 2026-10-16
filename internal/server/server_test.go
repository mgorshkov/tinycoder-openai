package server

import (
	"bufio"
	"context"
	"encoding/json"
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
	cfg.AuthToken = ""
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
	cfg.AuthToken = "secret"
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
