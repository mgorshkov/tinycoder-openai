package server

import (
	"context"
	"io"
	"log/slog"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mgorshkov/tinycoder-openai/internal/config"
	"github.com/mgorshkov/tinycoder-openai/internal/engine"
)

// overlapEngine reports, through shared counters, how many Generate calls are
// inside any engine at once. It stands in for the native engine's single-tenant
// state (one process-wide thread pool, one GPU device).
type overlapEngine struct {
	*engine.Stub
	cur, peak *int32
	delay     time.Duration
	entered   chan<- struct{}
}

func (e *overlapEngine) Generate(ctx context.Context, req engine.GenerateRequest, onToken engine.TokenFunc) (engine.Usage, error) {
	n := atomic.AddInt32(e.cur, 1)
	for {
		p := atomic.LoadInt32(e.peak)
		if n <= p || atomic.CompareAndSwapInt32(e.peak, p, n) {
			break
		}
	}
	if e.entered != nil {
		select {
		case e.entered <- struct{}{}:
		default:
		}
	}
	time.Sleep(e.delay)
	atomic.AddInt32(e.cur, -1)
	return engine.Usage{}, nil
}

// Two resident models with max_concurrent_requests = 2 must still execute one
// generation at a time: the semaphore admits both, Manager.exec keeps the engine
// single-tenant. Without that gate the second request enters the engine while
// the first is still inside it.
func TestManagerExecutesOneGenerationAcrossModels(t *testing.T) {
	cfg := config.Default()
	cfg.MaxLoadedModels = 2
	cfg.MaxConcurrent = 2
	cfg.Models = []config.RegistryModel{
		{ID: "alpha", Name: "Alpha", File: "alpha.gguf", ContextLength: 4096},
		{ID: "beta", Name: "Beta", File: "beta.gguf", ContextLength: 4096},
	}

	var cur, peak int32
	entered := make(chan struct{}, 2)
	factory := func(id string) engine.Engine {
		return &overlapEngine{
			Stub:    engine.NewStub(id),
			cur:     &cur,
			peak:    &peak,
			delay:   100 * time.Millisecond,
			entered: entered,
		}
	}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	mgr := NewManager(cfg, factory, log)

	run := func(id string) *sync.WaitGroup {
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := mgr.Generate(context.Background(), id,
				engine.GenerateRequest{Prompt: "hi", MaxTokens: 4}, nil); err != nil {
				t.Errorf("generate %q: %v", id, err)
			}
		}()
		return &wg
	}

	// Start "alpha", wait until it is inside the engine, then start "beta" while
	// alpha is still generating.
	wgA := run("alpha")
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("alpha never entered the engine")
	}
	wgB := run("beta")
	wgA.Wait()
	wgB.Wait()

	if got := atomic.LoadInt32(&peak); got != 1 {
		t.Fatalf("%d generations overlapped inside the engine, want 1", got)
	}
}

// recordEngine captures the LoadOptions the manager used for its model.
type recordEngine struct {
	*engine.Stub
	opts engine.LoadOptions
}

func (e *recordEngine) Load(ctx context.Context, opts engine.LoadOptions) error {
	e.opts = opts
	return e.Stub.Load(ctx, opts)
}

// A model entry may size its own CPU pool; without an entry it inherits the
// global n_threads.
func TestManagerUsesPerModelThreads(t *testing.T) {
	cfg := config.Default()
	cfg.MaxLoadedModels = 2
	cfg.MaxConcurrent = 1
	cfg.NThreads = 6
	cfg.Models = []config.RegistryModel{
		{ID: "alpha", Name: "Alpha", File: "alpha.gguf", NThreads: 2},
		{ID: "beta", Name: "Beta", File: "beta.gguf"},
	}

	engines := map[string]*recordEngine{}
	factory := func(id string) engine.Engine {
		e := &recordEngine{Stub: engine.NewStub(id)}
		engines[id] = e
		return e
	}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	mgr := NewManager(cfg, factory, log)

	for _, id := range []string{"alpha", "beta"} {
		if _, err := mgr.Generate(context.Background(), id,
			engine.GenerateRequest{Prompt: "hi", MaxTokens: 2}, nil); err != nil {
			t.Fatalf("generate %q: %v", id, err)
		}
	}

	if got := engines["alpha"].opts.NThreads; got != 2 {
		t.Fatalf("alpha loaded with %d threads, want the per-model override 2", got)
	}
	if got := engines["beta"].opts.NThreads; got != 6 {
		t.Fatalf("beta loaded with %d threads, want the global default 6", got)
	}
}
