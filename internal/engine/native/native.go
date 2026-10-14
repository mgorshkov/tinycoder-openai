//go:build !tinycoder_stub

// Package native implements engine.Engine on top of the tinycoder-inference
// C++ engine through a small C ABI shim (tinycoder_shim.{h,cpp}). It is
// compiled by default; exclude it with the `tinycoder_stub` build tag (a stub
// binary carries no CGO dependency). CGO must be enabled and the shim built
// into libtinycoder_bridge (see CMakeLists.txt and scripts/build.sh).
package native

/*
#cgo CFLAGS: -I${SRCDIR}/../../../native
#cgo LDFLAGS: -L${SRCDIR}/../../../build/native -Wl,-rpath,${SRCDIR}/../../../build/native -ltinycoder_bridge
#include <stdlib.h>
#include "tinycoder_shim.h"

// Provided by the //exports in callback.go.
extern int tcGoTokenCallback(uintptr_t, int32_t, const char*);
extern int tcGoAbortPredicate(uintptr_t);

// Adapt the Go trampolines into C function pointers of the shim's types.
// Declared in this file because they contain no //export directive.
static tc_token_cb tcGoTokenCallbackPtr(void) { return tcGoTokenCallback; }
static tc_abort_cb tcGoAbortPredicatePtr(void) { return tcGoAbortPredicate; }
*/
import "C"

import (
	"context"
	"fmt"
	"runtime"
	"runtime/cgo"
	"time"
	"unsafe"

	"github.com/mgorshkov/tinycoder-openai/internal/engine"
)

// Engine is the native engine.Engine implementation.
type Engine struct {
	modelID string
	handle  C.tc_engine_t
	loaded  bool
	info    engine.ModelInfo
}

// New creates a native engine bound to modelID. The C++ model is constructed
// immediately but no weights are read until Load is called.
func New(modelID string) *Engine {
	e := &Engine{modelID: modelID}
	e.handle = C.tc_engine_new()
	return e
}

func lastError() string {
	return C.GoString(C.tc_last_error())
}

// Load loads the GGUF model through the shim.
func (e *Engine) Load(ctx context.Context, opts engine.LoadOptions) error {
	if e.handle == nil {
		e.handle = C.tc_engine_new()
		if e.handle == nil {
			return fmt.Errorf("%w: could not construct engine: %s", engine.ErrUnavailable, lastError())
		}
	}
	// Configure the shared CPU thread pool BEFORE loading so the configured
	// n_threads (TINYCODER_THREADS / config) engages multiple workers. Without
	// this the pool is never created and every parallel region runs serially on
	// one core. 0 lets the engine choose (all logical CPUs).
	C.tc_engine_set_threads(e.handle, C.int(opts.NThreads))
	cPath := C.CString(opts.ModelPath)
	defer C.free(unsafe.Pointer(cPath))
	if rc := C.tc_engine_load(e.handle, cPath, C.int(opts.NThreads), C.int(opts.MaxSeqLen)); rc != 0 {
		return fmt.Errorf("load %q: %s", opts.ModelPath, lastError())
	}
	e.loaded = true
	e.info = e.queryInfo()
	return nil
}

// Unload releases the model and the C++ object.
func (e *Engine) Unload() error {
	if e.handle == nil {
		return nil
	}
	rc := C.tc_engine_unload(e.handle)
	e.handle = nil
	e.loaded = false
	e.info = engine.ModelInfo{}
	if rc != 0 {
		return fmt.Errorf("unload: %s", lastError())
	}
	return nil
}

// Loaded reports whether a model is resident.
func (e *Engine) Loaded() bool { return e.loaded }

// Info returns metadata for the loaded model.
func (e *Engine) Info() engine.ModelInfo { return e.info }

// Capabilities reports native capability flags.
func (e *Engine) Capabilities() engine.Capabilities {
	return engine.Capabilities{Native: true, Streaming: true, Chat: true}
}

func (e *Engine) queryInfo() engine.ModelInfo {
	var (
		name    [256]C.char
		arch    [128]C.char
		vocab   C.int
		layers  C.int
		hidden  C.int
		heads   C.int
		kvHeads C.int
		maxSeq  C.int
	)
	C.tc_engine_info(e.handle, &name[0], C.int(len(name)), &arch[0], C.int(len(arch)),
		&vocab, &layers, &hidden, &heads, &kvHeads, &maxSeq)
	return engine.ModelInfo{
		ID:            e.modelID,
		Name:          C.GoString(&name[0]),
		Architecture:  C.GoString(&arch[0]),
		NumLayers:     int(layers),
		HiddenSize:    int(hidden),
		NumHeads:      int(heads),
		NumKVHeads:    int(kvHeads),
		VocabSize:     int(vocab),
		MaxSeqLen:     int(maxSeq),
		ContextLength: int(maxSeq),
	}
}

// formatChat renders messages through the model's own chat template.
func (e *Engine) formatChat(messages []engine.Message, addGenerationPrompt bool) (string, error) {
	n := len(messages)
	roles := make([]*C.char, n)
	contents := make([]*C.char, n)
	for i, m := range messages {
		roles[i] = C.CString(m.Role)
		contents[i] = C.CString(m.Content)
	}
	defer func() {
		for i := 0; i < n; i++ {
			C.free(unsafe.Pointer(roles[i]))
			C.free(unsafe.Pointer(contents[i]))
		}
	}()
	var rolesPtr, contentsPtr *(*C.char)
	if n > 0 {
		rolesPtr = (**C.char)(unsafe.Pointer(&roles[0]))
		contentsPtr = (**C.char)(unsafe.Pointer(&contents[0]))
	}
	// First call with no buffer to learn the required length.
	length := C.tc_engine_format_chat(e.handle, rolesPtr, contentsPtr, C.int(n),
		boolToInt(addGenerationPrompt), nil, 0)
	if length < 0 {
		return "", fmt.Errorf("format chat: %s", lastError())
	}
	buf := make([]byte, int(length)+1)
	C.tc_engine_format_chat(e.handle, rolesPtr, contentsPtr, C.int(n),
		boolToInt(addGenerationPrompt), (*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf)))
	return string(buf[:length]), nil
}

// countPromptTokens tokenizes text to report prompt token usage.
func (e *Engine) countPromptTokens(text string) int {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	n := C.tc_engine_tokenize(e.handle, cText, nil, 0)
	if n < 0 {
		return 0
	}
	return int(n)
}

// Generate runs the model and streams tokens through req-specific callback.
//
// The C++ engine runs on its own OS thread (cgo.LockOSThread). If the caller's
// context is canceled (client disconnect) or the hard timeout elapses, the
// abort predicate (polled inside the engine's per-layer / per-block loop)
// requests a cooperative stop. Crucially, this method then WAITS for the engine
// call to actually return before it does, so the manager's per-model
// serialization stays valid: the model lock and the concurrency slot are only
// released once no engine code is running. Abandoning a live call — freeing or
// reusing the engine while tc_engine_generate is still on the stack — races the
// engine's C++ state (KV cache, buffers) and crashes the process with a
// SIGSEGV.
//
// The engine clears its own KV cache at the start of every generate() and the
// cooperative-abort path here restores it (clearKV), so the loaded model is
// reused for the next request without a reload.
func (e *Engine) Generate(ctx context.Context, req engine.GenerateRequest, onToken engine.TokenFunc) (engine.Usage, error) {
	var usage engine.Usage
	if !e.loaded {
		return usage, engine.ErrNotLoaded
	}

	prompt := req.Prompt
	if len(req.Messages) > 0 {
		rendered, err := e.formatChat(req.Messages, true)
		if err != nil {
			return usage, err
		}
		prompt = rendered
	}
	usage.PromptTokens = e.countPromptTokens(prompt)

	cPrompt := C.CString(prompt)

	gc := newGenCtx(ctx, onToken, req)
	h := cgo.NewHandle(gc)

	cb := C.tcGoTokenCallbackPtr()
	abort := C.tcGoAbortPredicatePtr()

	// The engine handle is only read here and is never cleared while the call is
	// in flight, so the worker always operates on a valid handle.
	handle := e.handle

	done := make(chan struct{})
	var genErr error
	var generated int
	go func() {
		// The worker owns cPrompt and the cgo handle for the whole call; they are
		// only freed once the C call has returned.
		defer C.free(unsafe.Pointer(cPrompt))
		defer h.Delete()
		// Pin the goroutine so the engine occupies its own thread; CGO calls
		// already lock the OS thread for their duration.
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		defer close(done)
		rc := C.tc_engine_generate(
			handle, cPrompt,
			C.int(req.MaxTokens), C.float(req.Temperature), C.float(req.TopP),
			C.float(req.TopK), C.float(req.RepeatPenalty), C.int(req.RepeatLastN),
			C.uint32_t(req.Seed), cb, abort, C.uintptr_t(h),
		)
		if rc < 0 {
			// A negative rc is the cooperative abort the shim returns when its
			// layer loop stops early. Only treat it as a hard error when neither
			// the caller's context, the hard deadline, the no-token guard, nor
			// the callback asked us to stop.
			if gc.canceled.Load() || gc.noToken.Load() || gc.stopped.Load() {
				generated = int(gc.tokens.Load())
				return
			}
			genErr = fmt.Errorf("generate: %s", lastError())
			return
		}
		generated = int(rc)
	}()

	// The hard timer is nil (never fires) when HardTimeout <= 0; in that case
	// ctx (the manager's deadline) is the only external bound.
	var hardC <-chan time.Time
	if req.HardTimeout > 0 {
		timer := time.NewTimer(req.HardTimeout)
		defer timer.Stop()
		hardC = timer.C
	}

	// finalize converts a completed worker result into the return value. It must
	// only be called after the worker has returned.
	finalize := func() (engine.Usage, error) {
		usage.CompletionTokens = generated
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
		switch {
		case genErr != nil:
			return usage, genErr
		case gc.canceled.Load():
			return usage, engine.ErrContextCanceled
		case gc.noToken.Load():
			return usage, engine.ErrNoToken
		default:
			return usage, nil
		}
	}

	// awaitWorker blocks until the engine call returns, then clears the KV cache
	// so a cooperatively aborted generation cannot corrupt the next request. It
	// always returns because the abort predicate is polled in the engine's
	// per-layer / per-block loop.
	awaitWorker := func() {
		<-done
		_ = e.clearKV()
	}

	select {
	case <-done:
		return finalize()
	case <-ctx.Done():
		// Prefer a completed result over a simultaneous cancel.
		select {
		case <-done:
			return finalize()
		default:
		}
		awaitWorker()
		return finalize()
	case <-hardC:
		select {
		case <-done:
			return finalize()
		default:
		}
		awaitWorker()
		usage.CompletionTokens = generated
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
		return usage, fmt.Errorf("%w: generation exceeded hard timeout", engine.ErrContextCanceled)
	}
}

// clearKV clears the model's KV / recurrent state (best-effort). The engine also
// clears it at the start of every generate(), so a failure here is not fatal.
func (e *Engine) clearKV() error {
	if e.handle == nil {
		return nil
	}
	if rc := C.tc_engine_clear_kv(e.handle); rc != 0 {
		return fmt.Errorf("clear kv: %s", lastError())
	}
	return nil
}

func boolToInt(b bool) C.int {
	if b {
		return 1
	}
	return 0
}
