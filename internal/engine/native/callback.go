//go:build tinycoder_native

package native

/*
#include <stdint.h>
*/
import "C"

import (
	"context"
	"runtime/cgo"

	"github.com/mgorshkov/tinycoder-openai/internal/engine"
)

// genCtx carries the per-call Go state referenced from the C callback via a
// cgo.Handle (an integer), which avoids passing Go pointers into C.
type genCtx struct {
	ctx      context.Context
	onToken  engine.TokenFunc
	canceled bool
	stopped  bool
}

//export tcGoTokenCallback
func tcGoTokenCallback(user C.uintptr_t, tokenID C.int32_t, text *C.char) C.int {
	h := cgo.Handle(uintptr(user))
	gc, ok := h.Value().(*genCtx)
	if !ok || gc == nil {
		return 1
	}
	select {
	case <-gc.ctx.Done():
		gc.canceled = true
		return 1
	default:
	}
	if gc.onToken != nil {
		if !gc.onToken(engine.Token{ID: int32(tokenID), Text: C.GoString(text)}) {
			gc.stopped = true
			return 1
		}
	}
	return 0
}
