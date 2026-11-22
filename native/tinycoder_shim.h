/*
 * tinycoder_shim.h — C ABI over the tinycoder-inference C++ engine.
 *
 * This header is the single boundary crossed via CGO. It intentionally uses
 * only plain C types (no C++ classes, no std::string) so the Go side never
 * needs to understand C++ ABI details.
 */
#ifndef TINYCODER_SHIM_H
#define TINYCODER_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque engine handle. */
typedef void *tc_engine_t;

/* Bump this when the ABI changes incompatibly.
 * 2: tc_engine_generate takes an abort predicate + threaded-thread setter. */
#define TC_ABI_VERSION 2

/* Construct a new (empty) C++ Model and return an opaque handle.
 * The handle must be released with tc_engine_unload. */
tc_engine_t tc_engine_new(void);

/* Load a GGUF model. nThreads<=0 means auto; maxSeqLen<=0 means model default.
 * Returns 0 on success, non-zero on failure (see tc_last_error). */
int tc_engine_load(tc_engine_t engine, const char *model_path, int n_threads,
                   int max_seq_len);

/* Release the loaded model. Returns 0 on success. */
int tc_engine_unload(tc_engine_t engine);

/* Query model metadata. Any out pointer may be NULL. Strings are written to the
 * caller-provided buffers (truncated, always NUL-terminated when cap>0). */
void tc_engine_info(tc_engine_t engine, char *name, int name_cap,
                    char *architecture, int arch_cap, int *vocab_size,
                    int *num_layers, int *hidden_size, int *num_heads,
                    int *num_kv_heads, int *max_seq_len);

/* Render a chat prompt with the model's template.
 * roles[i] and contents[i] are UTF-8 NUL-terminated strings.
 * On success writes the rendered prompt into out (NUL-terminated, up to cap)
 * and returns the full length (which may exceed cap-1). On failure returns a
 * negative value (see tc_last_error). */
int tc_engine_format_chat(tc_engine_t engine, const char *const *roles,
                          const char *const *contents, int count,
                          int add_generation_prompt, char *out, int cap);

/* Tokenize text, returning the number of tokens; token ids are written to out
 * (up to cap). Returns a negative value on error. */
int tc_engine_tokenize(tc_engine_t engine, const char *text, int32_t *out,
                       int cap);

/* Streaming callback invoked per generated token. Return 0 to continue, non-0
 * to stop generation early. `text` is a UTF-8 NUL-terminated string owned by
 * the engine and valid only for the duration of the call. `user` is an opaque
 * integer token (the Go side passes a cgo.Handle as a uintptr_t, so no Go
 * pointer ever crosses the boundary). */
typedef int (*tc_token_cb)(uintptr_t user, int32_t token_id, const char *text);

/* Cooperative abort predicate, polled by the engine between per-layer and
 * per-block units of work (NOT only per generated token). Return non-0 to
 * request a stop. Cheap and non-blocking; `user` is an opaque cgo.Handle. NULL
 * disables the check. */
typedef int (*tc_abort_cb)(uintptr_t user);

/* Set the number of CPU worker threads used by the shared thread pool.
 * n_threads<=0 restores the engine default (all logical CPUs). The pool is
 * created once, so the FIRST call wins. Returns 0 on success. */
int tc_engine_set_threads(tc_engine_t engine, int n_threads);

/* Clear the engine's KV / recurrent state (a no-op for a not-loaded model).
 * Used to restore a pristine cache after a cooperatively aborted generation.
 * Returns 0 on success. */
int tc_engine_clear_kv(tc_engine_t engine);

/* Run generation. Prompt is the fully rendered prompt string.
 * `abort_cb` (may be NULL) is polled inside the per-layer loop.
 * Returns the number of generated tokens (>=0) on success, or -1 on error AND
 * on a cooperative abort (callers should treat -1 as "aborted" when their own
 * predicate reported cancellation). */
int tc_engine_generate(tc_engine_t engine, const char *prompt, int max_tokens,
                       float temperature, float top_p, float top_k,
                       float repeat_penalty, int repeat_last_n, uint32_t seed,
                       tc_token_cb cb, tc_abort_cb abort_cb, uintptr_t user);

/* Thread-safe last error message for the calling thread. Never NULL. */
const char *tc_last_error(void);

/* Returns the ABI version compiled into the shim. */
int tc_abi_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TINYCODER_SHIM_H */
