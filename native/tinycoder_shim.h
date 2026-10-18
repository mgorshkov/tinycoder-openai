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

/* Bump this when the ABI changes incompatibly. */
#define TC_ABI_VERSION 1

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

/* Run generation. Prompt is the fully rendered prompt string.
 * Returns the number of generated tokens (>=0) on success, negative on error. */
int tc_engine_generate(tc_engine_t engine, const char *prompt, int max_tokens,
                       float temperature, float top_p, float top_k,
                       float repeat_penalty, int repeat_last_n, uint32_t seed,
                       tc_token_cb cb, uintptr_t user);

/* Thread-safe last error message for the calling thread. Never NULL. */
const char *tc_last_error(void);

/* Returns the ABI version compiled into the shim. */
int tc_abi_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TINYCODER_SHIM_H */
