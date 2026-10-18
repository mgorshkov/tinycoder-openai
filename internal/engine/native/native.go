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

// Provided by the //export in callback.go.
extern int tcGoTokenCallback(uintptr_t, int32_t, const char*);

// Adapts the Go trampoline into a C function pointer of the shim's type.
// Declared in this file because it contains no //export directive.
static tc_token_cb tcGoTokenCallbackPtr(void) { return tcGoTokenCallback; }
*/
import "C"

import (
	"context"
	"fmt"
	"runtime/cgo"
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

	gc := &genCtx{ctx: ctx, onToken: onToken}
	h := cgo.NewHandle(gc)
	defer h.Delete()

	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	cb := C.tcGoTokenCallbackPtr()
	rc := C.tc_engine_generate(
		e.handle, cPrompt,
		C.int(req.MaxTokens), C.float(req.Temperature), C.float(req.TopP),
		C.float(req.TopK), C.float(req.RepeatPenalty), C.int(req.RepeatLastN),
		C.uint32_t(req.Seed), cb, C.uintptr_t(h),
	)
	if rc < 0 {
		return usage, fmt.Errorf("generate: %s", lastError())
	}
	usage.CompletionTokens = int(rc)
	usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	if gc.canceled {
		return usage, engine.ErrContextCanceled
	}
	return usage, nil
}

func boolToInt(b bool) C.int {
	if b {
		return 1
	}
	return 0
}
