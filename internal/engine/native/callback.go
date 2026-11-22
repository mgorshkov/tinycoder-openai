//go:build !tinycoder_stub

package native

/*
#include <stdint.h>
*/
import "C"

import (
	"context"
	"runtime/cgo"
	"sync/atomic"
	"time"

	"github.com/mgorshkov/tinycoder-openai/internal/engine"
)

// genCtx carries the per-call Go state referenced from the C callback and the
// C abort predicate via a cgo.Handle (an integer), which avoids passing Go
// pointers into C.
type genCtx struct {
	ctx        context.Context
	onToken    engine.TokenFunc
	onProgress engine.ProgressFunc

	// canceled/stopped/noToken are read+written on the engine thread(s)
	// (callback and abort predicate) and read on the request goroutine after
	// tc_engine_generate returns, so they are atomics.
	//
	// canceled means the request context was done or the hard deadline passed;
	// noToken means the engine produced no token within noTokenTimeout; stopped
	// means the token callback asked to stop. They are distinguished so the HTTP
	// layer can report an accurate error (timeout vs no_token) instead of an
	// empty "generate failed".
	canceled atomic.Bool
	stopped  atomic.Bool
	noToken  atomic.Bool

	// lastActivity is the UnixNano of the last token, prefill-progress event
	// (or generation start), refreshed from the token callback and the abort
	// predicate. noTokenTimeout bounds the idle window while the engine is
	// running; hardDeadline bounds the whole call.
	lastActivity   atomic.Int64
	noTokenTimeout time.Duration
	hardDeadline   time.Time

	// tokens counts emitted tokens so the request goroutine can report partial
	// usage when a generation ends early (abort / no-token / hard timeout).
	tokens atomic.Int32
}

// newGenCtx builds the per-call state, seeding the activity clock so a hung
// engine is still bounded by noTokenTimeout.
func newGenCtx(ctx context.Context, onToken engine.TokenFunc, req engine.GenerateRequest) *genCtx {
	gc := &genCtx{
		ctx:            ctx,
		onToken:        onToken,
		onProgress:     req.OnProgress,
		noTokenTimeout: req.NoTokenTimeout,
	}
	gc.lastActivity.Store(time.Now().UnixNano())
	if req.HardTimeout > 0 {
		gc.hardDeadline = time.Now().Add(req.HardTimeout)
	}
	return gc
}

// noteProgress refreshes the idle clock; onProgress runs outside any lock.
func (gc *genCtx) noteProgress() {
	gc.lastActivity.Store(time.Now().UnixNano())
	if gc.onProgress != nil {
		gc.onProgress()
	}
}

// noTokenExpired reports whether the inactivity window has elapsed. A zero
// timeout disables the guard, and a genCtx that never recorded activity cannot
// expire (the clock is seeded when generation starts).
func (gc *genCtx) noTokenExpired(now time.Time) bool {
	if gc.noTokenTimeout <= 0 {
		return false
	}
	last := gc.lastActivity.Load()
	if last <= 0 {
		return false
	}
	return now.Sub(time.Unix(0, last)) > gc.noTokenTimeout
}

// shouldAbort is the cooperative abort predicate handed to the C engine. It is
// polled inside the engine's per-layer / per-block loop (not only per token).
//
// It returns true when the request context is done, the total wall-clock
// deadline has passed, or no token or progress has been produced within
// noTokenTimeout.
//
// A poll proves the engine is alive: it is polled between per-layer / per-block
// units of work during CPU prefill and partial-offload prefill, so each poll is
// treated as prefill progress and keeps the request alive.
//
//export tcGoAbortPredicate
func tcGoAbortPredicate(user C.uintptr_t) C.int {
	return C.int(abortPredicate(cgo.Handle(uintptr(user))))
}

// abortPredicate is the Go body of tcGoAbortPredicate, split out so tests can
// drive it with a cgo.Handle (Go does not support cgo in test files).
func abortPredicate(h cgo.Handle) int {
	gc, ok := h.Value().(*genCtx)
	if !ok || gc == nil {
		return 0
	}
	if gc.ctx != nil {
		select {
		case <-gc.ctx.Done():
			gc.canceled.Store(true)
			return 1
		default:
		}
	}
	// Refresh the idle clock (and forward the progress event) before the
	// timeout check, so a live engine cannot trip the no-token guard.
	gc.noteProgress()
	now := time.Now()
	if !gc.hardDeadline.IsZero() && now.After(gc.hardDeadline) {
		gc.canceled.Store(true)
		return 1
	}
	if gc.noTokenExpired(now) {
		// Record that the no-token guard (not a client disconnect) tripped,
		// so the caller can return engine.ErrNoToken and the HTTP layer can
		// surface code "no_token" rather than a generic error.
		gc.noToken.Store(true)
		return 1
	}
	return 0
}

//export tcGoTokenCallback
func tcGoTokenCallback(user C.uintptr_t, tokenID C.int32_t, text *C.char) C.int {
	h := cgo.Handle(uintptr(user))
	gc, ok := h.Value().(*genCtx)
	if !ok || gc == nil {
		return 1
	}
	// A token is already progress: refresh the idle clock and count it, but do
	// not call onProgress (the manager wraps onToken itself).
	gc.lastActivity.Store(time.Now().UnixNano())
	gc.tokens.Add(1)
	if gc.ctx != nil {
		select {
		case <-gc.ctx.Done():
			gc.canceled.Store(true)
			return 1
		default:
		}
	}
	if gc.onToken != nil {
		if !gc.onToken(engine.Token{ID: int32(tokenID), Text: C.GoString(text)}) {
			gc.stopped.Store(true)
			return 1
		}
	}
	return 0
}
