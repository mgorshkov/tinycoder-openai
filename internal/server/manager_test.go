package server

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/mgorshkov/tinycoder-openai/internal/engine"
)

// newWatchdogManager builds a Manager whose configured no-token timeout is
// disabled; these tests exercise the per-request value only.
func newWatchdogManager(t *testing.T, factory engine.Factory) *Manager {
	t.Helper()
	cfg := testConfig()
	cfg.NoTokenTimeoutSec = 0
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	return NewManager(cfg, factory, log)
}

// progressEngine simulates a long prefill: it keeps reporting engine progress
// through req.OnProgress and never emits a token.
type progressEngine struct {
	*engine.Stub
	interval time.Duration
	duration time.Duration
}

func (e *progressEngine) Generate(ctx context.Context, req engine.GenerateRequest, onToken engine.TokenFunc) (engine.Usage, error) {
	deadline := time.Now().Add(e.duration)
	for {
		if req.OnProgress != nil {
			req.OnProgress()
		}
		select {
		case <-ctx.Done():
			return engine.Usage{}, ctx.Err()
		case <-time.After(e.interval):
		}
		if time.Now().After(deadline) {
			return engine.Usage{}, nil
		}
	}
}

// hungEngine never emits a token and never reports progress. It returns
// engine.ErrContextCanceled as soon as the request context is canceled (the
// native engine's cooperative abort), or after block if block is positive (used
// to prove the guard can be disabled).
type hungEngine struct {
	*engine.Stub
	block time.Duration
}

func (e *hungEngine) Generate(ctx context.Context, req engine.GenerateRequest, onToken engine.TokenFunc) (engine.Usage, error) {
	if e.block > 0 {
		select {
		case <-ctx.Done():
			return engine.Usage{}, engine.ErrContextCanceled
		case <-time.After(e.block):
			return engine.Usage{}, nil
		}
	}
	<-ctx.Done()
	return engine.Usage{}, engine.ErrContextCanceled
}

// A long prefill that emits no token but keeps reporting engine progress must
// not trip the no-token watchdog: progress proves the engine is alive, even
// though the request then runs for five times the inactivity window.
func TestNoTokenWatchdogSurvivesPrefillProgress(t *testing.T) {
	const window = 60 * time.Millisecond
	mgr := newWatchdogManager(t, func(id string) engine.Engine {
		return &progressEngine{
			Stub:     engine.NewStub(id),
			interval: 10 * time.Millisecond,
			duration: 300 * time.Millisecond,
		}
	})

	start := time.Now()
	_, err := mgr.Generate(context.Background(), "alpha", engine.GenerateRequest{
		Prompt:         "hi",
		MaxTokens:      8,
		NoTokenTimeout: window,
	}, nil)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Generate returned %v after %s; a prefill that reports progress must not be aborted",
			err, elapsed)
	}
	// Prove the request really outlived the inactivity window by a wide margin,
	// so the test cannot pass trivially by finishing early.
	if elapsed <= 3*window {
		t.Fatalf("Generate returned after %s, want well past 3x the %s inactivity window", elapsed, window)
	}
}

// A hung engine that never reports progress must still be aborted with the
// existing no-token error, promptly and with the same message.
func TestNoTokenWatchdogAbortsHungEngine(t *testing.T) {
	mgr := newWatchdogManager(t, func(id string) engine.Engine {
		return &hungEngine{Stub: engine.NewStub(id)}
	})

	type result struct {
		err error
	}
	done := make(chan result, 1)
	start := time.Now()
	go func() {
		_, err := mgr.Generate(context.Background(), "alpha", engine.GenerateRequest{
			Prompt:         "hi",
			MaxTokens:      8,
			NoTokenTimeout: 60 * time.Millisecond,
		}, nil)
		done <- result{err: err}
	}()

	// Own deadline: a watchdog regression fails fast instead of hanging until
	// the Go test timeout.
	var err error
	select {
	case r := <-done:
		err = r.err
	case <-time.After(2 * time.Second):
		t.Fatal("Generate did not return within 2s; the inactivity watchdog failed to abort a hung engine")
	}
	elapsed := time.Since(start)
	if !errors.Is(err, engine.ErrNoToken) {
		t.Fatalf("Generate returned %v, want engine.ErrNoToken", err)
	}
	if elapsed > time.Second {
		t.Fatalf("hung engine aborted after %s, want well under a second", elapsed)
	}
}

// lateErrorEngine outlives the inactivity window (ignoring cancellation) and
// then returns a genuine engine failure.
type lateErrorEngine struct {
	*engine.Stub
	delay time.Duration
}

func (e *lateErrorEngine) Generate(ctx context.Context, req engine.GenerateRequest, onToken engine.TokenFunc) (engine.Usage, error) {
	time.Sleep(e.delay)
	return engine.Usage{}, errors.New("kernel launched a bad grid")
}

// A genuine engine error must win over a simultaneously tripped watchdog; only
// a nil/canceled result is rewritten to ErrNoToken.
func TestNoTokenWatchdogDoesNotMaskEngineError(t *testing.T) {
	mgr := newWatchdogManager(t, func(id string) engine.Engine {
		return &lateErrorEngine{Stub: engine.NewStub(id), delay: 120 * time.Millisecond}
	})

	start := time.Now()
	_, err := mgr.Generate(context.Background(), "alpha", engine.GenerateRequest{
		Prompt:         "hi",
		MaxTokens:      8,
		NoTokenTimeout: 30 * time.Millisecond,
	}, nil)
	if err == nil || !strings.Contains(err.Error(), "bad grid") {
		t.Fatalf("Generate returned %v, want the engine's own error", err)
	}
	if errors.Is(err, engine.ErrNoToken) {
		t.Fatalf("Generate returned %v, want the genuine engine error, not ErrNoToken", err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("Generate returned after %s, want well under a second", elapsed)
	}
}

// A zero timeout disables the guard entirely: a request that reports no
// progress for longer than any window still completes normally.
func TestNoTokenWatchdogDisabledWhenZero(t *testing.T) {
	mgr := newWatchdogManager(t, func(id string) engine.Engine {
		return &hungEngine{Stub: engine.NewStub(id), block: 150 * time.Millisecond}
	})

	start := time.Now()
	_, err := mgr.Generate(context.Background(), "alpha", engine.GenerateRequest{
		Prompt:         "hi",
		MaxTokens:      8,
		NoTokenTimeout: 0,
	}, nil)
	if err != nil {
		t.Fatalf("Generate returned %v with the guard disabled, want nil", err)
	}
	if elapsed := time.Since(start); elapsed < 150*time.Millisecond {
		t.Fatalf("Generate returned after %s, want the engine's 150ms", elapsed)
	}
}
