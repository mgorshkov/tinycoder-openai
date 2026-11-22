package server

import (
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

// preloadTestConfig is a one-model configuration with the startup warm-up
// enabled and a budget generous enough for the gated-engine tests.
func preloadTestConfig() config.Config {
	cfg := config.Default()
	cfg.Models = []config.RegistryModel{
		{ID: "alpha", Name: "Alpha", File: "alpha.gguf", ContextLength: 4096},
	}
	cfg.Preload = true
	cfg.PreloadTimeoutSec = 5
	return cfg
}

// newPreloadServer builds a Server around an injected engine factory so the
// warm-up can be driven deterministically, and exposes the Manager for
// eviction-level assertions.
func newPreloadServer(t *testing.T, cfg config.Config, factory engine.Factory) (*Server, *httptest.Server) {
	t.Helper()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	mgr := NewManager(cfg, factory, log)
	srv := New(cfg, mgr, log)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return srv, ts
}

type readyzBody struct {
	Status string           `json:"status"`
	Models []ModelReadiness `json:"models"`
}

func getReadyz(t *testing.T, baseURL string) (int, readyzBody) {
	t.Helper()
	resp, err := http.Get(baseURL + "/readyz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "application/json") {
		t.Fatalf("/readyz Content-Type = %q, want application/json", ct)
	}
	var body readyzBody
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode /readyz: %v", err)
	}
	return resp.StatusCode, body
}

// waitReadyz polls /readyz until it reports want, so the tests do not race the
// background warm-up goroutine.
func waitReadyz(t *testing.T, baseURL, want string) (int, readyzBody) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	var (
		code int
		body readyzBody
	)
	for {
		code, body = getReadyz(t, baseURL)
		if body.Status == want {
			return code, body
		}
		if time.Now().After(deadline) {
			t.Fatalf("/readyz status = %q (HTTP %d), want %q within 5s", body.Status, code, want)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func getHealthz(t *testing.T, baseURL string) int {
	t.Helper()
	resp, err := http.Get(baseURL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	return resp.StatusCode
}

// gatedEngine blocks Load until release is closed; a nil release blocks until
// the load context ends.
type gatedEngine struct {
	*engine.Stub
	started chan struct{}
	release chan struct{}
	once    sync.Once
}

func (e *gatedEngine) Load(ctx context.Context, opts engine.LoadOptions) error {
	e.once.Do(func() { close(e.started) })
	select {
	case <-e.release:
		return e.Stub.Load(ctx, opts)
	case <-ctx.Done():
		return ctx.Err()
	}
}

// ctxIgnoringEngine blocks Load until release is closed and ignores the context
// entirely, modeling the native engine's blocking C load.
type ctxIgnoringEngine struct {
	*engine.Stub
	started  chan struct{}
	release  chan struct{}
	returned chan struct{}
	once     sync.Once
}

func (e *ctxIgnoringEngine) Load(ctx context.Context, opts engine.LoadOptions) error {
	e.once.Do(func() { close(e.started) })
	<-e.release
	close(e.returned)
	return e.Stub.Load(ctx, opts)
}

// failingEngine fails its first `failures` Load calls, then succeeds.
type failingEngine struct {
	*engine.Stub
	failures atomic.Int32
}

// newFailingEngine returns an engine whose first failures Load calls fail.
func newFailingEngine(id string, failures int32) *failingEngine {
	e := &failingEngine{Stub: engine.NewStub(id)}
	e.failures.Store(failures)
	return e
}

func (e *failingEngine) Load(ctx context.Context, opts engine.LoadOptions) error {
	if e.failures.Load() > 0 {
		e.failures.Add(-1)
		return errors.New("boom: cannot open model file")
	}
	return e.Stub.Load(ctx, opts)
}

// A server with no preload targets has nothing to warm: readiness is immediate
// and, like /healthz, requires no API key.
func TestReadyzNoTargetsIsReady(t *testing.T) {
	cfg := config.Default()
	cfg.Models = []config.RegistryModel{{ID: "alpha", Name: "Alpha", File: "alpha.gguf"}}
	cfg.APIKeys = []string{"sk-test"} // /readyz must stay unauthenticated
	_, ts := newPreloadServer(t, cfg, func(id string) engine.Engine { return engine.NewStub(id) })

	code, body := getReadyz(t, ts.URL)
	if code != http.StatusOK {
		t.Fatalf("/readyz status = %d, want 200", code)
	}
	if body.Status != "ready" {
		t.Fatalf("/readyz status = %q, want ready", body.Status)
	}
	if len(body.Models) != 0 {
		t.Fatalf("/readyz models = %+v, want none", body.Models)
	}
	if code := getHealthz(t, ts.URL); code != http.StatusOK {
		t.Fatalf("/healthz status = %d, want 200", code)
	}
}

// While the warm-up is blocked in Load, /readyz reports loading (503) and
// /healthz stays 200: liveness must not fail merely because the model is not
// warm yet. Releasing the load flips readiness to 200 ready.
func TestReadyzLoadingWhileWarmupBlocks(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseLoad := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseLoad)

	cfg := preloadTestConfig()
	srv, ts := newPreloadServer(t, cfg, func(id string) engine.Engine {
		return &gatedEngine{Stub: engine.NewStub(id), started: started, release: release}
	})
	srv.StartPreload()

	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("warm-up never started")
	}

	code, body := getReadyz(t, ts.URL)
	if code != http.StatusServiceUnavailable {
		t.Fatalf("/readyz status while loading = %d, want 503", code)
	}
	if body.Status != "loading" {
		t.Fatalf("/readyz status while loading = %q, want loading", body.Status)
	}
	if len(body.Models) != 1 || body.Models[0].ID != "alpha" || body.Models[0].Loaded || body.Models[0].Error != "" {
		t.Fatalf("/readyz models while loading = %+v, want alpha not loaded and no error", body.Models)
	}
	if code := getHealthz(t, ts.URL); code != http.StatusOK {
		t.Fatalf("/healthz status while loading = %d, want 200 (liveness only)", code)
	}

	releaseLoad()
	code, body = waitReadyz(t, ts.URL, "ready")
	if code != http.StatusOK {
		t.Fatalf("/readyz status after warm-up = %d, want 200", code)
	}
	if len(body.Models) != 1 || body.Models[0].ID != "alpha" || !body.Models[0].Loaded || body.Models[0].Error != "" {
		t.Fatalf("/readyz models after warm-up = %+v, want alpha loaded with no error", body.Models)
	}
}

// A failed warm-up is reported as 503 error with the engine message, while
// /healthz remains 200.
func TestReadyzErrorWhenWarmupFails(t *testing.T) {
	cfg := preloadTestConfig()
	srv, ts := newPreloadServer(t, cfg, func(id string) engine.Engine {
		return newFailingEngine(id, 1)
	})
	srv.StartPreload()

	code, body := waitReadyz(t, ts.URL, "error")
	if code != http.StatusServiceUnavailable {
		t.Fatalf("/readyz status after a failed warm-up = %d, want 503", code)
	}
	if len(body.Models) != 1 || body.Models[0].ID != "alpha" || body.Models[0].Loaded {
		t.Fatalf("/readyz models after a failed warm-up = %+v, want alpha not loaded", body.Models)
	}
	if !strings.Contains(body.Models[0].Error, "boom") {
		t.Fatalf("/readyz model error = %q, want the engine's message", body.Models[0].Error)
	}
	if code := getHealthz(t, ts.URL); code != http.StatusOK {
		t.Fatalf("/healthz status after a failed warm-up = %d, want 200", code)
	}
}

// A load that never finishes is bounded by preload_timeout_sec: readiness turns
// to 503 error mentioning the timeout after about the configured budget.
func TestReadyzWarmupTimeout(t *testing.T) {
	cfg := preloadTestConfig()
	cfg.PreloadTimeoutSec = 1
	started := make(chan struct{})
	srv, ts := newPreloadServer(t, cfg, func(id string) engine.Engine {
		// Nil release: Load blocks until the warm-up budget cancels it.
		return &gatedEngine{Stub: engine.NewStub(id), started: started}
	})

	start := time.Now()
	srv.StartPreload()
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("warm-up never started")
	}

	code, body := waitReadyz(t, ts.URL, "error")
	elapsed := time.Since(start)
	if code != http.StatusServiceUnavailable {
		t.Fatalf("/readyz status after a timed-out warm-up = %d, want 503", code)
	}
	if len(body.Models) != 1 || body.Models[0].Loaded {
		t.Fatalf("/readyz models after a timed-out warm-up = %+v", body.Models)
	}
	if elapsed < 900*time.Millisecond {
		t.Fatalf("warm-up reported failure after %s, want about the 1s budget", elapsed)
	}
	if elapsed > 3*time.Second {
		t.Fatalf("warm-up reported failure after %s, want about the 1s budget", elapsed)
	}
	msg := strings.ToLower(body.Models[0].Error)
	if !strings.Contains(msg, "timed out") || !strings.Contains(msg, "1s") {
		t.Fatalf("/readyz model error = %q, want it to name the timeout", body.Models[0].Error)
	}
}

// A native model load is a blocking C call that cannot be canceled. The budget
// must still be reported: /readyz turns to 503 error with the timeout message
// once preload_timeout_sec has elapsed even while Load is still blocked, and
// flips to 200 ready when the load finally completes.
func TestReadyzWarmupTimeoutWhileLoadIgnoresContext(t *testing.T) {
	cfg := preloadTestConfig()
	cfg.PreloadTimeoutSec = 1
	started := make(chan struct{})
	release := make(chan struct{})
	returned := make(chan struct{})
	var releaseOnce sync.Once
	releaseLoad := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseLoad)

	srv, ts := newPreloadServer(t, cfg, func(id string) engine.Engine {
		return &ctxIgnoringEngine{
			Stub:     engine.NewStub(id),
			started:  started,
			release:  release,
			returned: returned,
		}
	})

	start := time.Now()
	srv.StartPreload()
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("warm-up never started")
	}

	code, body := waitReadyz(t, ts.URL, "error")
	elapsed := time.Since(start)
	if code != http.StatusServiceUnavailable {
		t.Fatalf("/readyz status after the budget expired = %d, want 503", code)
	}
	if elapsed < 900*time.Millisecond {
		t.Fatalf("warm-up reported the timeout after %s, want about the 1s budget", elapsed)
	}
	if elapsed > 3*time.Second {
		t.Fatalf("warm-up reported the timeout after %s, want about the 1s budget", elapsed)
	}
	if len(body.Models) != 1 || body.Models[0].Loaded {
		t.Fatalf("/readyz models after the budget expired = %+v, want alpha not loaded", body.Models)
	}
	msg := strings.ToLower(body.Models[0].Error)
	if !strings.Contains(msg, "timed out") || !strings.Contains(msg, "1s") {
		t.Fatalf("/readyz model error = %q, want it to name the timeout", body.Models[0].Error)
	}
	// The engine load must still be blocked: the timeout is reported, not
	// enforced by abandoning the call.
	select {
	case <-returned:
		t.Fatal("Load returned before release; the timeout must be reporting-only")
	default:
	}
	if code := getHealthz(t, ts.URL); code != http.StatusOK {
		t.Fatalf("/healthz while the load is blocked = %d, want 200", code)
	}

	releaseLoad()
	code, body = waitReadyz(t, ts.URL, "ready")
	if code != http.StatusOK {
		t.Fatalf("/readyz after the load completed = %d, want 200", code)
	}
	if len(body.Models) != 1 || !body.Models[0].Loaded || body.Models[0].Error != "" {
		t.Fatalf("/readyz models after the load completed = %+v, want alpha loaded with no error", body.Models)
	}
}

// Readiness is sticky on success: a later idle eviction must not flip a warmed
// process back to 503, because the model is still loadable and the process is
// healthy.
func TestReadyzStickyAfterEviction(t *testing.T) {
	cfg := preloadTestConfig()
	srv, ts := newPreloadServer(t, cfg, func(id string) engine.Engine { return engine.NewStub(id) })
	srv.StartPreload()

	if _, body := waitReadyz(t, ts.URL, "ready"); len(body.Models) != 1 || !body.Models[0].Loaded {
		t.Fatalf("/readyz models after warm-up = %+v, want alpha loaded", body.Models)
	}
	mgr := srv.mgr
	if !mgr.IsLoaded("alpha") {
		t.Fatal("warm-up did not leave alpha resident")
	}

	// Evict exactly as the idle loop reserves a victim, then unload it.
	mgr.mu.Lock()
	e := mgr.entries["alpha"]
	e.evicting = true
	mgr.mu.Unlock()
	mgr.unload(e)
	if mgr.IsLoaded("alpha") {
		t.Fatal("unload left alpha loaded")
	}

	code, body := getReadyz(t, ts.URL)
	if code != http.StatusOK || body.Status != "ready" {
		t.Fatalf("/readyz after an idle eviction = %d %q, want 200 ready (sticky)", code, body.Status)
	}
	if len(body.Models) != 1 || !body.Models[0].Loaded {
		t.Fatalf("/readyz models after an idle eviction = %+v, want alpha still reported warm", body.Models)
	}
}

// A lazy request that loads a model whose warm-up failed repairs readiness: any
// successful Acquire records the model as warm and clears the stored error.
func TestReadyzRepairedByLazyLoad(t *testing.T) {
	cfg := preloadTestConfig()
	eng := newFailingEngine("alpha", 1)
	srv, ts := newPreloadServer(t, cfg, func(id string) engine.Engine { return eng })
	srv.StartPreload()

	if _, body := waitReadyz(t, ts.URL, "error"); body.Models[0].Error == "" {
		t.Fatal("/readyz did not report the failed warm-up")
	}

	resp := postJSON(t, ts.URL+"/v1/generate", `{"model":"alpha","prompt":"hi","max_tokens":4}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("lazy load request status = %d, body=%s", resp.StatusCode, raw)
	}

	code, body := getReadyz(t, ts.URL)
	if code != http.StatusOK || body.Status != "ready" {
		t.Fatalf("/readyz after a successful lazy load = %d %q, want 200 ready", code, body.Status)
	}
	if len(body.Models) != 1 || !body.Models[0].Loaded || body.Models[0].Error != "" {
		t.Fatalf("/readyz models after a successful lazy load = %+v, want alpha loaded with no error", body.Models)
	}
}

// /readyz is unauthenticated but, like the content endpoints, rejects non-GET
// with 405 in the existing JSON error shape.
func TestReadyzRejectsNonGET(t *testing.T) {
	cfg := preloadTestConfig()
	_, ts := newPreloadServer(t, cfg, func(id string) engine.Engine { return engine.NewStub(id) })

	resp, err := http.Post(ts.URL+"/readyz", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("POST /readyz status = %d, want 405", resp.StatusCode)
	}
	var eb errorBody
	if err := json.NewDecoder(resp.Body).Decode(&eb); err != nil {
		t.Fatal(err)
	}
	if eb.Error.Type != "invalid_request_error" || eb.Error.Code != "" {
		t.Fatalf("POST /readyz error = %+v, want invalid_request_error with no code", eb.Error)
	}
}
