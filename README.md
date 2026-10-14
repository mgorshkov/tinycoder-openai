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
- **Tool calling** (OpenAI `tools` / `tool_choice` / `tool_calls`), including
  streamed tool-call deltas and `finish_reason: "tool_calls"`, so agentic
  clients such as Roo Code can drive function calls.
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

All content endpoints require an API key when the server is configured with
`api_keys` (see [API keys](#api-keys)). Pass one as
`Authorization: Bearer <key>` (or `?api_key=<key>`):

```bash
# Your key, as listed under "api_keys" in the server config.
export TINYCODER_API_KEY="sk-live-abc123"

# List models
curl -s https://api.tinycoder.dev/v1/models \
  -H "Authorization: Bearer $TINYCODER_API_KEY" | jq

# Chat completion
curl -s https://api.tinycoder.dev/v1/chat/completions \
  -H "Authorization: Bearer $TINYCODER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "qwen3.6-35b-a3b",
        "messages": [{"role": "user", "content": "Write a Go hello world."}],
        "max_tokens": 256,
        "temperature": 0.7
      }' | jq

# Streaming
curl -N https://api.tinycoder.dev/v1/chat/completions \
  -H "Authorization: Bearer $TINYCODER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-coder-1.5b","messages":[{"role":"user","content":"hi"}],"stream":true}'

# The key may also be passed as a query parameter.
curl -s "https://api.tinycoder.dev/v1/models?api_key=$TINYCODER_API_KEY" | jq
```

Point any OpenAI SDK at it by setting the base URL to
`https://api.tinycoder.dev/v1` and using a model id from `/v1/models`, with the
API key as the SDK's `api_key` (sent as the `Authorization: Bearer` header).
The server renders chat templates itself, so send `messages` rather than a
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

The **native engine is the default**: a plain `go build` compiles it (with CGO
enabled) and links the bridge. The dependency-free stub is opt-in via the
`tinycoder_stub` build tag.

```bash
# Native engine with CUDA GPU offload (default)
./scripts/build.sh              # or: make build / make native

# Native engine, CPU only
./scripts/build.sh --cpu

# Stub engine (no native toolchain) — for development / tests
./scripts/build.sh --stub       # or: make stub
# ...or directly:
go build -tags tinycoder_stub -o build/bin/tinycoder-openai ./cmd/tinycoder-openai

# Tests (stub engine; no native toolchain required)
make test
```

The native build configures CMake (fetching `tinycoder-inference`, then its
`np`/`cmake` dependencies, on the first run), produces
`build/native/libtinycoder_bridge.so`, then compiles the Go binary with
`CGO_ENABLED=1` (no build tag). The cgo directive bakes in an rpath to
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
| `--auth-token` | `TINYCODER_AUTH_TOKEN` | *(none)* | Single bearer token (legacy; empty = open) |
| `--log-level` | `TINYCODER_LOG_LEVEL` | `info` | `debug`/`info`/`warn`/`error` |

Other environment variables: `TINYCODER_MAX_LOADED_MODELS`,
`TINYCODER_IDLE_TIMEOUT_SEC`, `TINYCODER_THREADS`, `TINYCODER_MAX_SEQ_LEN`,
`TINYCODER_DEFAULT_MAX_TOKENS`, `TINYCODER_MAX_PROMPT_TOKENS`,
`TINYCODER_REQUEST_TIMEOUT_SEC`, `TINYCODER_MAX_CONCURRENT`,
`TINYCODER_ALLOWED_ORIGINS` (comma-separated), `TINYCODER_API_KEYS`
(comma-separated), `TINYCODER_TOOL_CALLING`, `TINYCODER_SCAN_MODELS_DIR`.

### Tool calling

Tool calling is enabled by default (`"tool_calling": true`) and is available on
`POST /v1/chat/completions`. Send an OpenAI `tools` array (with optional
`tool_choice`) and the model's calls are returned as standard `tool_calls`.

Because the inference engine returns free text rather than constrained output,
calls are elicited through a text protocol (`<tool_call>{json}</tool_call>`)
described in a system message, and parsed back into structured `tool_calls`.
This works with any instruction-following chat model but, as a text protocol, is
best-effort: it depends on the model obeying the instructions.

- `tool_choice`: `"auto"` (default), `"none"`, `"required"`, or
  `{"type":"function","function":{"name":"..."}}`.
- Non-streaming: the assistant message carries `tool_calls` and
  `finish_reason` is `"tool_calls"`.
- Streaming: tool calls arrive as `delta.tool_calls` fragments (one per call,
  with `index`), and the terminal chunk has `finish_reason: "tool_calls"`.
- Multi-turn: echo assistant `tool_calls` back and send `tool` messages with
  `tool_call_id` and the result; they are rendered into the prompt.
- `GET /v1/models` advertises a `tools` capability when enabled.

Example:

```bash
curl -s https://api.tinycoder.dev/v1/chat/completions \
  -H "Authorization: Bearer $TINYCODER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "qwen3.6-35b-a3b",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": [{
          "type": "function",
          "function": {
            "name": "get_weather",
            "description": "Look up the current weather.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
          }
        }]
      }' | jq
```

### API keys

Authentication is enforced whenever at least one key is configured. Accepted
keys are the union of the `api_keys` list and the legacy single `auth_token`:

```json
{ "api_keys": ["sk-live-abc123", "sk-live-def456"] }
```

Clients must present a configured key as `Authorization: Bearer <key>` (or
`?api_key=<key>`). A missing or unknown key receives `401` with
`{"error":{"type":"authentication_error","code":"invalid_api_key"}}`. When no
keys are configured the endpoints are open (intended only for local use);
`/healthz` is always unauthenticated for liveness probes.

See [`config.example.json`](config.example.json) for the full schema and the
editable model registry. Registry entries map a public `id` to a GGUF `file`
(relative to `models_dir`, or absolute).

The config file is the **authoritative** model registry: `GET /v1/models`
advertises only the models declared under `"models"`, and no other model id is
accepted by the generation endpoints. Models present under `models_dir` but not
listed in `"models"` are never served, so experimental or unsupported GGUFs
(e.g. ones that OOM the GPU at load) cannot leak into the API. To publish a
model, add an entry to `"models"`.

`scan_models_dir` / `TINYCODER_SCAN_MODELS_DIR` only control
[`Config.DiscoverModels`](internal/config/config.go:263), a helper for tooling
that wants to enumerate on-disk GGUFs; its results are **not** merged into the
served registry.

## Deployment

See [`deploy/`](deploy/) for a systemd unit, an nginx reverse-proxy snippet
(with SSE buffering disabled) and a Dockerfile. The server exposes `/healthz`
for load-balancer health checks and streams SSE with
`X-Accel-Buffering: no` so nginx does not buffer token chunks.

### deploy.sh

[`scripts/deploy.sh`](scripts/deploy.sh:1) installs build artifacts to the paths
described by a JSON manifest (default
[`deploy/deploy.config.json`](deploy/deploy.config.json:1)). Each manifest entry
maps an artifact to a destination path, mode, owner and behaviour. Destination
paths may reference `${prefix}`, `${config_dir}` and `${systemd_dir}`, which are
resolved from the manifest (and overridable with `--prefix`).

| Manifest key | Meaning |
|--------------|---------|
| `prefix`, `config_dir`, `systemd_dir` | Base directories referenced as `${prefix}` etc. in destinations |
| `user`, `group` | Default owner applied to installed files |
| `artifacts[].source` / `.dest` | Artifact to copy (relative to the repo) and its destination |
| `artifacts[].mode` | File mode (octal) |
| `artifacts[].required` | Fail if the built artifact is missing |
| `artifacts[].if_missing` | Keep a pre-existing destination (e.g. an edited config) |
| `service.unit` / `.reload` / `.enable` / `.restart` | systemd unit to reload / enable / restart |

```bash
# Build, then install to the configured paths (needs root)
make build             # native engine (or: make stub)
sudo make deploy

# Preview what would change
make deploy-dry

# Stage a tree under DIR (no root needed) — handy for packaging
./scripts/deploy.sh --destdir /tmp/pkg
```

The binary lands in `${prefix}`, the native `libtinycoder_bridge.so` in
`${prefix}/lib` (matching the unit's `LD_LIBRARY_PATH`), the config seeds
`${config_dir}/config.json` only if absent, and the unit is copied to
`${systemd_dir}` and reloaded/enabled/restarted. `jq` or `python3` is required
to read the manifest; destination directories are created as needed.

## Security

- Set `api_keys` (or `TINYCODER_API_KEYS`, or the legacy `auth_token`) in any
  public deployment; the server then requires `Authorization: Bearer <key>`
  (or `?api_key=<key>`) on all content endpoints. Any configured key is
  accepted; unknown or missing keys get `401`.
- `allowed_origins` defaults to `*`; restrict it to your front-end origins if
  you expose the API to browsers.
- `max_prompt_tokens` bounds `max_tokens`; `request_timeout_sec` bounds every
  generation and cancels it on client disconnect.
