//go:build !tinycoder_stub

package native

import (
	"context"
	"runtime/cgo"
	"testing"
	"time"

	"github.com/mgorshkov/tinycoder-openai/internal/engine"
)

// TestNoTokenExpired pins the inactivity decision the abort predicate uses: the
// window is measured from the last activity (token or prefill progress), a zero
// timeout disables it, and a genCtx that never recorded activity never expires.
func TestNoTokenExpired(t *testing.T) {
	now := time.Now()
	tests := []struct {
		name    string
		timeout time.Duration
		last    time.Time // zero means "no activity recorded"
		want    bool
	}{
		{name: "inside window", timeout: 60 * time.Second, last: now.Add(-59 * time.Second), want: false},
		{name: "past window", timeout: 60 * time.Second, last: now.Add(-61 * time.Second), want: true},
		{name: "zero timeout disables", timeout: 0, last: now.Add(-time.Hour), want: false},
		{name: "no activity recorded", timeout: 60 * time.Second, last: time.Time{}, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gc := &genCtx{noTokenTimeout: tt.timeout}
			if !tt.last.IsZero() {
				gc.lastActivity.Store(tt.last.UnixNano())
			}
			if got := gc.noTokenExpired(now); got != tt.want {
				t.Fatalf("noTokenExpired() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestAbortPredicateReportsProgress pins the abort-predicate -> progress
// integration: every poll refreshes the idle clock and forwards a prefill
// progress event to the caller's OnProgress, while a canceled context aborts
// before any progress is reported.
func TestAbortPredicateReportsProgress(t *testing.T) {
	var progress int
	gc := newGenCtx(context.Background(), nil, engine.GenerateRequest{
		OnProgress: func() { progress++ },
	})
	h := cgo.NewHandle(gc)
	defer h.Delete()

	if got := abortPredicate(h); got != 0 {
		t.Fatalf("abortPredicate() = %d, want 0", got)
	}
	if progress != 1 {
		t.Fatalf("progress callback fired %d time(s) after one poll, want 1", progress)
	}
	if got := abortPredicate(h); got != 0 {
		t.Fatalf("abortPredicate() = %d on the second poll, want 0", got)
	}
	if progress != 2 {
		t.Fatalf("progress callback fired %d time(s) after two polls, want 2", progress)
	}

	// A canceled context must abort without reporting progress.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	gc = newGenCtx(ctx, nil, engine.GenerateRequest{
		OnProgress: func() { progress++ },
	})
	h = cgo.NewHandle(gc)
	defer h.Delete()
	before := progress
	if got := abortPredicate(h); got != 1 {
		t.Fatalf("abortPredicate() with a canceled context = %d, want 1", got)
	}
	if progress != before {
		t.Fatalf("progress callback fired on a canceled context (count %d -> %d)", before, progress)
	}
}
