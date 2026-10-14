/*
 * tinycoder_shim.cpp — implementation of the C ABI over tinycoder-inference.
 *
 * Compiled as C++20 and linked against the tinycoder_core static library.
 * All exceptions are caught at the boundary and converted to error codes so an
 * engine failure can never unwind into the Go runtime.
 */

#include "tinycoder_shim.h"

#include "Model.hpp"
#include "ModelConfig.hpp"
#include "ThreadPool.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <new>
#include <string>
#include <vector>

namespace {

// Per-thread last-error storage; the Go side reads it immediately after a
// failed call on the same OS thread.
thread_local std::string g_lastError;

void setError(const std::string &msg) {
    g_lastError = msg;
}

void clearError() {
    g_lastError.clear();
}

// Per-thread cooperative-abort state for the current tc_engine_generate call
// on this OS thread. The abort predicate is invoked between per-layer /
// per-block units of work from the generate thread (the token callback runs on
// the same thread), so thread_local is sufficient and lock-free.
thread_local tc_abort_cb g_abortCb = nullptr;
thread_local uintptr_t g_abortUser = 0;
thread_local bool g_aborted = false;

inline bool abortRequested() {
    if (g_aborted) {
        return true;
    }
    if (g_abortCb != nullptr && g_abortCb(g_abortUser) != 0) {
        g_aborted = true;
        return true;
    }
    return false;
}

// Copies src into dst (capacity cap), truncating and always NUL-terminating.
void copyOut(const std::string &src, char *dst, int cap) {
    if (dst == nullptr || cap <= 0) {
        return;
    }
    const size_t n = src.size() < static_cast<size_t>(cap - 1)
                             ? src.size()
                             : static_cast<size_t>(cap - 1);
    std::memcpy(dst, src.data(), n);
    dst[n] = '\0';
}

struct CallbackCtx {
    tc_token_cb cb = nullptr;
    uintptr_t user = 0;
    int count = 0;
    bool stopped = false;
};

}  // namespace

extern "C" {

tc_engine_t tc_engine_new(void) {
    clearError();
    try {
        return static_cast<tc_engine_t>(new tinycoder::Model());
    } catch (const std::exception &e) {
        setError(e.what());
        return nullptr;
    } catch (...) {
        setError("unknown exception during construction");
        return nullptr;
    }
}

int tc_abi_version(void) {
    return TC_ABI_VERSION;
}

const char *tc_last_error(void) {
    return g_lastError.c_str();
}

int tc_engine_load(tc_engine_t engine, const char *model_path, int n_threads,
                   int max_seq_len) {
    clearError();
    if (engine == nullptr) {
        setError("null engine handle");
        return 1;
    }
    if (model_path == nullptr || model_path[0] == '\0') {
        setError("empty model path");
        return 1;
    }
    auto *model = static_cast<tinycoder::Model *>(engine);
    try {
        auto &cfg = const_cast<tinycoder::ModelConfig &>(model->config());
        if (n_threads > 0) {
            cfg.nThreads = static_cast<uint32_t>(n_threads);
        }
        if (max_seq_len > 0) {
            cfg.maxSeqLen = static_cast<uint32_t>(max_seq_len);
        }
        std::string err;
        if (!model->load(std::string(model_path), &err, nullptr)) {
            setError(err.empty() ? "model load failed" : err);
            return 1;
        }

#ifdef USE_CUDA
        // Eager GPU offload: upload the (possibly partial) weights now so the
        // first request does not pay a multi-second/multi-GB upload stall, and
        // an offload that does not fit is reported at load time (falling back
        // to CPU) instead of surfacing as a slow or failed first token.
        // TINYCODER_GPU_EAGER=0 disables the eager upload.
        const char *eager = std::getenv("TINYCODER_GPU_EAGER");
        if (eager == nullptr || eager[0] != '0') {
            std::string gerr;
            if (model->gpuUploadIfEnabled(&gerr)) {
                std::cout << "[TinyCoder] GPU offload active: "
                          << model->gpuNGpuLayers() << " layer(s) resident on device"
                          << std::endl;
            } else if (!gerr.empty()) {
                std::cout << "[TinyCoder] GPU offload inactive (CPU path): "
                          << gerr << std::endl;
            }
        }
#endif
        return 0;
    } catch (const std::exception &e) {
        setError(e.what());
        return 1;
    } catch (...) {
        setError("unknown exception during load");
        return 1;
    }
}

int tc_engine_unload(tc_engine_t engine) {
    clearError();
    if (engine == nullptr) {
        return 0;
    }
    try {
        delete static_cast<tinycoder::Model *>(engine);
        return 0;
    } catch (const std::exception &e) {
        setError(e.what());
        return 1;
    } catch (...) {
        setError("unknown exception during unload");
        return 1;
    }
}

void tc_engine_info(tc_engine_t engine, char *name, int name_cap,
                    char *architecture, int arch_cap, int *vocab_size,
                    int *num_layers, int *hidden_size, int *num_heads,
                    int *num_kv_heads, int *max_seq_len) {
    clearError();
    if (engine == nullptr) {
        return;
    }
    auto *model = static_cast<tinycoder::Model *>(engine);
    const tinycoder::ModelConfig &cfg = model->config();
    copyOut(cfg.modelName, name, name_cap);
    copyOut(cfg.architecture, architecture, arch_cap);
    if (vocab_size) *vocab_size = static_cast<int>(cfg.vocabSize);
    if (num_layers) *num_layers = static_cast<int>(cfg.numLayers);
    if (hidden_size) *hidden_size = static_cast<int>(cfg.hiddenSize);
    if (num_heads) *num_heads = static_cast<int>(cfg.numAttentionHeads);
    if (num_kv_heads) *num_kv_heads = static_cast<int>(cfg.numKVHeads);
    if (max_seq_len) *max_seq_len = static_cast<int>(cfg.maxSeqLen);
}

int tc_engine_format_chat(tc_engine_t engine, const char *const *roles,
                          const char *const *contents, int count,
                          int add_generation_prompt, char *out, int cap) {
    clearError();
    if (engine == nullptr) {
        setError("null engine handle");
        return -1;
    }
    try {
        auto *model = static_cast<tinycoder::Model *>(engine);
        std::vector<std::pair<std::string, std::string>> messages;
        messages.reserve(static_cast<size_t>(count > 0 ? count : 0));
        for (int i = 0; i < count; ++i) {
            const char *r = (roles && roles[i]) ? roles[i] : "user";
            const char *c = (contents && contents[i]) ? contents[i] : "";
            messages.emplace_back(std::string(r), std::string(c));
        }
        const std::string rendered =
                model->formatChat(messages, add_generation_prompt != 0);
        copyOut(rendered, out, cap);
        return static_cast<int>(rendered.size());
    } catch (const std::exception &e) {
        setError(e.what());
        return -1;
    } catch (...) {
        setError("unknown exception during format_chat");
        return -1;
    }
}

int tc_engine_tokenize(tc_engine_t engine, const char *text, int32_t *out,
                       int cap) {
    clearError();
    if (engine == nullptr) {
        setError("null engine handle");
        return -1;
    }
    try {
        auto *model = static_cast<tinycoder::Model *>(engine);
        const std::vector<int32_t> tokens =
                model->tokenize(text == nullptr ? "" : text);
        const int n = static_cast<int>(tokens.size());
        const int m = n < cap ? n : cap;
        for (int i = 0; i < m; ++i) {
            out[i] = tokens[static_cast<size_t>(i)];
        }
        return n;
    } catch (const std::exception &e) {
        setError(e.what());
        return -1;
    } catch (...) {
        setError("unknown exception during tokenize");
        return -1;
    }
}

int tc_engine_set_threads(tc_engine_t engine, int n_threads) {
    clearError();
    (void) engine;
    try {
        // The persistent pool is a process-wide singleton created once; the
        // first call after process start wins. This is the ONLY place the pool
        // is initialized, so before it every parallelFor ran serially (0%
        // multi-core utilisation).
        auto &pool = tinycoder::ThreadPool::instance();
        if (!pool.isInitialized()) {
            if (n_threads > 0) {
                pool.setAffinityEnabled(false);
                pool.initialize(static_cast<size_t>(n_threads));
            } else {
                pool.initialize(tinycoder::ThreadPool::recommendedThreadCount());
            }
        }
        return 0;
    } catch (const std::exception &e) {
        setError(e.what());
        return 1;
    } catch (...) {
        setError("unknown exception during set_threads");
        return 1;
    }
}

int tc_engine_clear_kv(tc_engine_t engine) {
    clearError();
    if (engine == nullptr) {
        return 0;
    }
    try {
        static_cast<tinycoder::Model *>(engine)->clearKVCache();
        return 0;
    } catch (const std::exception &e) {
        setError(e.what());
        return 1;
    } catch (...) {
        setError("unknown exception during clear_kv");
        return 1;
    }
}

int tc_engine_generate(tc_engine_t engine, const char *prompt, int max_tokens,
                       float temperature, float top_p, float top_k,
                       float repeat_penalty, int repeat_last_n, uint32_t seed,
                       tc_token_cb cb, tc_abort_cb abort_cb, uintptr_t user) {
    clearError();
    if (engine == nullptr) {
        setError("null engine handle");
        return -1;
    }
    auto *model = static_cast<tinycoder::Model *>(engine);
    try {
        tinycoder::InferenceParams params;
        if (max_tokens > 0) params.maxTokens = max_tokens;
        params.temperature = temperature;
        params.topP = top_p;
        params.topK = top_k;
        params.repeatPenalty = repeat_penalty;
        params.repeatLastN = repeat_last_n;
        params.seed = seed;

        CallbackCtx ctx;
        ctx.cb = cb;
        ctx.user = user;

        // Arm the per-thread abort predicate the engine polls inside its
        // per-layer / per-block loop, and always clear it on the way out.
        g_abortCb = abort_cb;
        g_abortUser = user;
        g_aborted = false;
        struct AbortReset {
            ~AbortReset() {
                g_abortCb = nullptr;
                g_abortUser = 0;
                g_aborted = false;
            }
        } abortReset;
        (void) abortReset;

        // Note: stop is cooperative. Model::generate clears the KV cache at
        // entry, so serializing requests per model (the load manager does this)
        // is sufficient for correctness.
        auto callback = [&ctx](int32_t tokenId, const std::string &text) -> bool {
            ctx.count++;
            if (g_abortCb != nullptr && abortRequested()) {
                ctx.stopped = true;
                return false;
            }
            if (ctx.cb != nullptr) {
                if (ctx.cb(ctx.user, tokenId, text.c_str()) != 0) {
                    ctx.stopped = true;
                    return false;
                }
            }
            return true;
        };

        // Temporarily bind the abort predicate into the per-block check value
        // consumed by the engine core (see Model::generate / forward).
        model->armAbortPredicate(&abortRequested);
        struct AbortUnbind {
            tinycoder::Model *m;
            ~AbortUnbind() { m->armAbortPredicate(nullptr); }
        } abortUnbind{model};
        (void) abortUnbind;

        const std::vector<int32_t> generated = model->generate(
                prompt == nullptr ? "" : std::string(prompt), params, callback);
        if (abortRequested()) {
            // Signal a cooperative abort (the Go side distinguishes this from a
            // hard error by consulting its own predicate).
            return -1;
        }
        return static_cast<int>(generated.size());
    } catch (const std::exception &e) {
        setError(e.what());
        return -1;
    } catch (...) {
        setError("unknown exception during generate");
        return -1;
    }
}

}  // extern "C"
