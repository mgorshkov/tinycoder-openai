# TinyCoder OpenAI Provider

An **OpenAI-compatible HTTP gateway** for the
[`tinycoder-inference`](https://github.com/mgorshkov/tinycoder-inference) engine.
It lets users point any OpenAI-style client at the public URL
`https://api.tinycoder.dev/v1`, pick a model from the registry
(`GET /v1/models`), and run chat or text completions — all served by the same
high-performance C++/CUDA GGUF engine the TinyCoder VS Code extension uses
locally.

```
┌─────────────┐   OpenAI-style HTTP/SSE   ┌──────────────────────┐   C ABI    ┌────────────────────────┐
│  Client /   │ ────────────────────────► │  tinycoder-openai    │ ─────────► │  tinycoder-inference   │
│  extension  │ ◄──────────────────────── │  (Go HTTP server)    │            │  (C++/CUDA engine)     │
└─────────────┘   JSON / text/event-stream └──────────────────────┘            └────────────────────────┘
```

## Features

- **OpenAI-compatible** `POST /v1/chat/completions`, `POST /v1/completions` and
  a native `POST /v1/generate`.
- **Model discovery** via `GET /v1/models`, with an OpenAI-shaped
  `{object: "list", data: [...]}` payload extended with tinycoder metadata
  (architecture, quantization, context length, loaded flag, capabilities).
- **Server-Sent Events streaming** (`"stream": true`) with OpenAI
  `chat.completion.chunk` / `text_completion` framing and a final
  `data: [DONE]`.
- **Chat templates** rendered inside the engine with the model's own GGUF
  template — the Go layer only forwards `messages`.
- **Lazy loading with LRU idle eviction**: models are loaded on first use and
  unloaded after an idle timeout; resident models are capped by
  `max_loaded_models`.
- **Per-model request serialization**: the engine uses a single KV cache, so
  concurrent requests for the same model are queued.
- **OpenAPI 3.1 spec** served at `/openai.json` and `/openai.yaml`.
- **Optional bearer-token auth**, configurable CORS, request timeouts and a
  global concurrency limit.
- **Two engine modes**: a native CGO bridge over the real engine, and a
  dependency-free **stub engine** for development, tests and HTTP-only
  deployments.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/v1/models` | List models |
| `POST` | `/v1/chat/completions` | Chat completion (streaming supported) |
| `POST` | `/v1/completions` | Text completion (streaming supported) |
| `POST` | `/v1/generate` | Native endpoint (accepts `prompt` **or** `messages`) |
| `GET`  | `/openai.json`, `/openai.yaml` | OpenAPI 3.1 document |
| `GET`  | `/healthz` | Liveness probe |

The base URL is `https://api.tinycoder.dev/v1`, so all content endpoints live
under the `/v1` prefix (e.g. `/v1/models`, `/v1/chat/completions`). Only these
paths are served — there are no unversioned or legacy aliases.

### Quick start

```bash
# List models
curl -s https://api.tinycoder.dev/v1/models | jq

# Chat completion
curl -s https://api.tinycoder.dev/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "qwen3.6-35b-a3b",
        "messages": [{"role": "user", "content": "Write a Go hello world."}],
        "max_tokens": 256,
        "temperature": 0.7
      }' | jq

# Streaming
curl -N https://api.tinycoder.dev/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-coder-1.5b","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Point any OpenAI SDK at it by setting the base URL to
`https://api.tinycoder.dev/v1` and using a model id from `/v1/models`. The
server renders chat templates itself, so send `messages` rather than a
pre-formatted prompt.

## Architecture

| Layer | Package | Responsibility |
|-------|---------|----------------|
| HTTP/OpenAPI | [`internal/server`](internal/server/server.go:1) | Routing, DTOs, SSE streaming, auth, CORS, spec |
| Model lifecycle | [`internal/server`](internal/server/manager.go:1) | Lazy load, LRU idle eviction, per-model serialization, concurrency |
| Engine contract | [`internal/engine`](internal/engine/engine.go:1) | Backend-agnostic `Engine` interface + stub |
| Native bridge | [`internal/engine/native`](internal/engine/native/native.go:1) | CGO binding to the C ABI shim |
| C ABI shim | [`native/tinycoder_shim.cpp`](native/tinycoder_shim.cpp:1) | `extern "C"` wrapper over the C++ `tinycoder::Model` |
| Wiring | [`internal/wiring`](internal/wiring/wiring_stub.go:1) | Build-tag selection of native vs stub engine |
| Config | [`internal/config`](internal/config/config.go:1) | JSON file + env + flags, model registry |

### Request flow

1. HTTP handler validates and normalizes the request
   ([`handlers.go`](internal/server/handlers.go:96)).
2. [`Manager.Generate`](internal/server/manager.go:296) acquires a global
   concurrency slot, then a per-model lease (loading the model if needed and
   evicting an idle LRU victim when at capacity).
3. The lease's [`Engine.Generate`](internal/engine/engine.go:120) runs the
   model; tokens stream through a callback.
4. For the native engine the callback crosses into C++ via a `cgo.Handle` and
   the shim's `tc_token_cb`; the token text is copied back to Go.
5. Streaming responses write SSE chunks and flush; non-streaming responses
   aggregate the text and emit a single JSON object with usage.

### Concurrency model

- One `engine.Engine` instance per registry model.
- `entry.mu` serializes load/unload/generate for a model — required because the
  C++ engine owns a single KV cache and clears it per request.
- `Manager.sem` caps total in-flight generations (`max_concurrent_requests`).
- The idle loop unloads models idle longer than `idle_timeout_sec`.
- With `max_loaded_models = N`, the manager guarantees at most `N` resident
  models by evicting the least-recently-used idle one when loading another.

## Build

Requirements: **Go ≥ 1.22**. Native builds additionally need a **C++20
compiler**, **CMake ≥ 3.18** and **Git**. `tinycoder-inference` is fetched
automatically from git via CMake `FetchContent` (the same way the engine
fetches its `np`/`cmake` dependencies); a specific revision can be pinned with
`-DTINYCODER_INFERENCE_TAG=<tag>`.

```bash
# Native engine with CUDA GPU offload (default)
./scripts/build.sh              # or: make native

# Native engine, CPU only
./scripts/build.sh --cpu

# Stub engine (no native toolchain) — for development / tests
./scripts/build.sh --stub       # or: make build

# Tests
make test
```

The native build configures CMake (fetching `tinycoder-inference`, then its
`np`/`cmake` dependencies, on the first run), produces
`build/native/libtinycoder_bridge.so`, then compiles the Go binary with the
`tinycoder_native` build tag. The cgo directive bakes in an rpath to
`build/native/`; for distribution, install the `.so` next to the binary or set
`LD_LIBRARY_PATH`.

CUDA is **on by default** (`-DENABLE_CUDA=ON`); pass `--cpu` /
`-DENABLE_CUDA=OFF` for a CPU-only build. AVX-512, AVX-512 VBMI/VBMI2 and AMX
are off by default (portable AVX2 build) because a base engine compiled with
AVX-512 will SIGILL on hosts that lack it — enable them with
`-DENABLE_AVX512=ON`, `-DENABLE_AVX512_VBMI=ON`, `-DENABLE_AVX512_VBMI2=ON`
and/or `-DENABLE_AMX=ON` only on known-capable hosts (e.g. Ice Lake+ for
VBMI/VBMI2, Sapphire Rapids for AMX, alongside a matching
`-DCMAKE_CUDA_ARCHITECTURES`).

## Configuration

Precedence: **flags > environment > JSON file > built-in defaults**.

```bash
./build/bin/tinycoder-openai \
  --config /etc/tinycoder/config.json \
  --listen :8080 \
  --models-dir /data/models \
  --auth-token "$TINYCODER_AUTH_TOKEN"
```

| Flag | Env | Default | Description |
|------|-----|---------|-------------|
| `--config` | `TINYCODER_CONFIG` | — | JSON config path |
| `--listen` | `TINYCODER_LISTEN` | `:8080` | Listen address |
| `--models-dir` | `TINYCODER_MODELS_DIR` | `/data/models` | GGUF root |
| `--auth-token` | `TINYCODER_AUTH_TOKEN` | *(none)* | Bearer token (empty = open) |
| `--log-level` | `TINYCODER_LOG_LEVEL` | `info` | `debug`/`info`/`warn`/`error` |

Other environment variables: `TINYCODER_MAX_LOADED_MODELS`,
`TINYCODER_IDLE_TIMEOUT_SEC`, `TINYCODER_THREADS`, `TINYCODER_MAX_SEQ_LEN`,
`TINYCODER_DEFAULT_MAX_TOKENS`, `TINYCODER_MAX_PROMPT_TOKENS`,
`TINYCODER_REQUEST_TIMEOUT_SEC`, `TINYCODER_MAX_CONCURRENT`,
`TINYCODER_ALLOWED_ORIGINS` (comma-separated), `TINYCODER_SCAN_MODELS_DIR`.

See [`config.example.json`](config.example.json) for the full schema and the
editable model registry. Registry entries map a public `id` to a GGUF `file`
(relative to `models_dir`, or absolute).

## Deployment

See [`deploy/`](deploy/) for a systemd unit, an nginx reverse-proxy snippet
(with SSE buffering disabled) and a Dockerfile. The server exposes `/healthz`
for load-balancer health checks and streams SSE with
`X-Accel-Buffering: no` so nginx does not buffer token chunks.

## Security

- Set `auth_token` (or `TINYCODER_AUTH_TOKEN`) in any public deployment; the
  server then requires `Authorization: Bearer <token>` (or `?api_key=`) on all
  content endpoints.
- `allowed_origins` defaults to `*`; restrict it to your front-end origins if
  you expose the API to browsers.
- `max_prompt_tokens` bounds `max_tokens`; `request_timeout_sec` bounds every
  generation and cancels it on client disconnect.
