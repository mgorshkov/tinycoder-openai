package server

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mgorshkov/tinycoder-openai/internal/engine"
)

// cancelEngine simulates an engine whose generation is aborted (client
// disconnect / no-token guard / hard timeout).
type cancelEngine struct{ *engine.Stub }

func (e *cancelEngine) Generate(ctx context.Context, req engine.GenerateRequest, onToken engine.TokenFunc) (engine.Usage, error) {
	if !e.Loaded() {
		return engine.Usage{}, engine.ErrNotLoaded
	}
	return engine.Usage{}, engine.ErrContextCanceled
}

// noTokenEngine simulates the no-token guard tripping.
type noTokenEngine struct{ *engine.Stub }

func (e *noTokenEngine) Generate(ctx context.Context, req engine.GenerateRequest, onToken engine.TokenFunc) (engine.Usage, error) {
	if !e.Loaded() {
		return engine.Usage{}, engine.ErrNotLoaded
	}
	return engine.Usage{}, engine.ErrNoToken
}

func newEngineServer(t *testing.T, mk func(id string) engine.Engine) *httptest.Server {
	t.Helper()
	cfg := testConfig()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	mgr := NewManager(cfg, mk, log)
	ts := httptest.NewServer(New(cfg, mgr, log).Handler())
	t.Cleanup(ts.Close)
	return ts
}

// TestGenerateCanceledReturnsGatewayTimeout verifies that a canceled generation
// is reported as a non-200 (504) rather than a silently truncated 200.
func TestGenerateCanceledReturnsGatewayTimeout(t *testing.T) {
	ts := newEngineServer(t, func(id string) engine.Engine {
		return &cancelEngine{Stub: engine.NewStub(id)}
	})
	body := `{"model":"alpha","prompt":"hi","max_tokens":8}`
	resp, err := http.Post(ts.URL+"/v1/generate", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusGatewayTimeout {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 504; body=%s", resp.StatusCode, b)
	}
	var eb errorBody
	if err := decodeJSON(resp.Body, &eb); err != nil {
		t.Fatal(err)
	}
	if eb.Error.Code != "timeout" {
		t.Fatalf("unexpected error code: %+v", eb)
	}
}

// TestGenerateNoTokenReturnsGatewayTimeout verifies the no-token guard surfaces
// as a non-200 (504, code "no_token").
func TestGenerateNoTokenReturnsGatewayTimeout(t *testing.T) {
	ts := newEngineServer(t, func(id string) engine.Engine {
		return &noTokenEngine{Stub: engine.NewStub(id)}
	})
	body := `{"model":"alpha","prompt":"hi","max_tokens":8}`
	resp, err := http.Post(ts.URL+"/v1/generate", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusGatewayTimeout {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 504; body=%s", resp.StatusCode, b)
	}
	var eb errorBody
	if err := decodeJSON(resp.Body, &eb); err != nil {
		t.Fatal(err)
	}
	if eb.Error.Code != "no_token" {
		t.Fatalf("unexpected error code: %+v", eb)
	}
}

// TestCanceledGenerationReleasesCapacity verifies that a canceled request does
// not leak its concurrency slot: a following request must still succeed.
func TestCanceledGenerationReleasesCapacity(t *testing.T) {
	cfg := testConfig()
	cfg.MaxConcurrent = 1
	log := slog.New(slog.NewTextHandler(io.Discard, nil))

	var calls int
	mk := func(id string) engine.Engine {
		return &countingCancelEngine{Stub: engine.NewStub(id), calls: &calls}
	}
	mgr := NewManager(cfg, mk, log)
	ts := httptest.NewServer(New(cfg, mgr, log).Handler())
	t.Cleanup(ts.Close)

	// First request: cancellation on the first call.
	body := `{"model":"alpha","prompt":"hi","max_tokens":8}`
	resp, err := http.Post(ts.URL+"/v1/generate", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusGatewayTimeout {
		t.Fatalf("first status = %d, want 504", resp.StatusCode)
	}

	// Second request: must acquire the single slot again and succeed.
	resp2, err := http.Post(ts.URL+"/v1/generate", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp2.Body)
		t.Fatalf("second status = %d, want 200 (slot leaked?); body=%s", resp2.StatusCode, b)
	}
}

// countingCancelEngine cancels the first Generate call and serves the rest.
type countingCancelEngine struct {
	*engine.Stub
	calls *int
}

func (e *countingCancelEngine) Generate(ctx context.Context, req engine.GenerateRequest, onToken engine.TokenFunc) (engine.Usage, error) {
	*e.calls++
	if *e.calls == 1 {
		return engine.Usage{}, engine.ErrContextCanceled
	}
	return e.Stub.Generate(ctx, req, onToken)
}
