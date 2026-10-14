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
	ctx     context.Context
	onToken engine.TokenFunc

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

	// lastActivity is the UnixNano of the last token (or generation start),
	// refreshed from the callback. noTokenTimeout bounds the idle window while
	// the engine is running; hardDeadline bounds the whole call.
	lastActivity   atomic.Int64
	noTokenTimeout time.Duration
	hardDeadline   time.Time

	// tokens counts emitted tokens so the request goroutine can report partial
	// usage when a generation ends early (abort / no-token / hard timeout).
	tokens atomic.Int32
}

// newGenCtx builds the per-call state, seeding the activity clock so a hung
// prefill (which emits no token) is still bounded by noTokenTimeout.
func newGenCtx(ctx context.Context, onToken engine.TokenFunc, req engine.GenerateRequest) *genCtx {
	gc := &genCtx{ctx: ctx, onToken: onToken, noTokenTimeout: req.NoTokenTimeout}
	gc.lastActivity.Store(time.Now().UnixNano())
	if req.HardTimeout > 0 {
		gc.hardDeadline = time.Now().Add(req.HardTimeout)
	}
	return gc
}

// shouldAbort is the cooperative abort predicate handed to the C engine. It is
// polled inside the engine's per-layer / per-block loop (not only per token).
//
// It returns true when the request context is done, the total wall-clock
// deadline has passed, or no token has been produced within noTokenTimeout.
//
//export tcGoAbortPredicate
func tcGoAbortPredicate(user C.uintptr_t) C.int {
	h := cgo.Handle(uintptr(user))
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
	now := time.Now()
	if !gc.hardDeadline.IsZero() && now.After(gc.hardDeadline) {
		gc.canceled.Store(true)
		return 1
	}
	if gc.noTokenTimeout > 0 {
		last := gc.lastActivity.Load()
		if last > 0 && now.Sub(time.Unix(0, last)) > gc.noTokenTimeout {
			// Record that the no-token guard (not a client disconnect) tripped,
			// so the caller can return engine.ErrNoToken and the HTTP layer can
			// surface code "no_token" rather than a generic error.
			gc.noToken.Store(true)
			return 1
		}
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
	// A token is activity: refresh the idle clock, count it, and honor context
	// cancellation.
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
