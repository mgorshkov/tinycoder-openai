# TinyCoder OpenAI Provider

An **OpenAI- and Anthropic-compatible HTTP gateway** for the
[`tinycoder-inference`](https://github.com/mgorshkov/tinycoder-inference) engine.
It lets users point any OpenAI-style or Anthropic-style client at the public URL
`https://api.tinycoder.dev/v1`, pick a model from the registry
(`GET /v1/models`), and run chat, text, Responses or Messages completions — all
served by the same high-performance C++/CUDA GGUF engine the TinyCoder VS Code
extension uses locally.

```
┌────────────────────────┐  OpenAI + Anthropic APIs   ┌────────────────────────┐    C ABI    ┌────────────────────────┐
│   Client / extension   │ ─────────────────────────► │    tinycoder-openai    │ ──────────► │  tinycoder-inference   │
│ OpenAI / Anthropic SDK │ ◄───────────────────────── │    (Go HTTP server)    │             │   (C++/CUDA engine)    │
└────────────────────────┘  JSON / text/event-stream  └────────────────────────┘             └────────────────────────┘
```

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

## Deployment

See [`deploy/`](deploy/) for a systemd unit, an nginx reverse-proxy snippet
(with SSE buffering disabled) and a Dockerfile. The server exposes `/healthz`
for liveness and `/readyz` for readiness checks; the snippet proxies both probe
paths explicitly so they bypass its catch-all, and the server streams SSE with
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

## Features

- **OpenAI-compatible** `POST /v1/chat/completions`, `POST /v1/completions` and
  `POST /v1/responses` (the OpenAI Responses API), plus `POST /v1/messages`
  (the Anthropic Messages API) and a native `POST /v1/generate`.
- **Model discovery** via `GET /v1/models`, with an OpenAI-shaped
  `{object: "list", data: [...]}` payload extended with tinycoder metadata
  (architecture, quantization, context length, loaded flag, capabilities).
- **Server-Sent Events streaming** (`"stream": true`) with OpenAI
  `chat.completion.chunk` / `text_completion` framing and a final
  `data: [DONE]` (the Responses and Messages endpoints stream their native
  typed events — `response.*` and `message_start` … `message_stop` — instead).
  While the model is loading or the prompt is being prefilled
  (which can take a while for large prompts and no tokens are produced yet), the
  server emits SSE keep-alive comment lines (`: keep-alive`) so clients do not
  hit their idle read timeout; a generation failure is surfaced instead of a
  silently truncated stream: `/v1/chat/completions` and `/v1/completions` emit
  an SSE `error` event followed by `[DONE]`, `/v1/responses` terminates with
  `response.failed` carrying `response.error{code,message}`, and `/v1/messages`
  uses an Anthropic `error` event.
- **Tool calling** (OpenAI `tools` / `tool_choice` / `tool_calls`), including
  streamed tool-call deltas and `finish_reason: "tool_calls"`, so agentic
  clients such as Roo Code can drive function calls.
- **Chat templates** rendered inside the engine with the model's own GGUF
  template — the Go layer only forwards `messages`.
- **Lazy loading with LRU idle eviction**: models are loaded on first use and
  unloaded after an idle timeout; resident models are capped by
  `max_loaded_models`.
- **Readiness and optional preload**: `/healthz` is liveness-only; `/readyz`
  accounts for the optional startup warm-up (`preload` / `preload_models`), so
  an orchestrator does not route traffic before the model is warm.
- **Per-model request serialization**: the engine uses a single KV cache, so
  concurrent requests for the same model are queued. `max_concurrent_requests`
  defaults to one slot per registered model and bounds admission (queue depth);
  the engine itself still executes one request at a time.
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
| `POST` | `/v1/responses` | OpenAI Responses API (streaming supported) |
| `POST` | `/v1/messages` | Anthropic Messages API (streaming supported) |
| `POST` | `/v1/generate` | Native endpoint (accepts `prompt` **or** `messages`) |
| `GET`  | `/openai.json`, `/openai.yaml` | OpenAPI 3.1 document |
| `GET`  | `/healthz` | Liveness probe (process up) |
| `GET`  | `/readyz` | Readiness probe (model warm-up) |

The base URL is `https://api.tinycoder.dev/v1`, so all content endpoints live
under the `/v1` prefix (e.g. `/v1/models`, `/v1/chat/completions`). Only these
paths are served — there are no unversioned or legacy aliases.

A prompt longer than the model's context window is refused with HTTP `400`. On
the OpenAI-compatible endpoints (`/v1/completions`, `/v1/chat/completions`,
`/v1/responses` and `/v1/generate`) the body is the OpenAI error shape,
`{"error":{"type":"invalid_request_error","code":"context_length_exceeded"}}`.
The Anthropic-shaped `POST /v1/messages` instead returns
`{"type":"error","error":{"type":"invalid_request_error","message":"..."}}`,
which carries no `code` field. Streaming requests preflight the prompt only when
the model is already resident and idle, so a cold or busy stream surfaces the
same refusal as an SSE `error` event (OpenAI-shaped on the OpenAI-compatible
streams, Anthropic-shaped on `/v1/messages`) rather than an empty success with
`finish_reason: "stop"`.

### Health and readiness

`/healthz` is a liveness probe: it returns `200` for as long as the process
serves HTTP, regardless of whether any model is loaded. `/readyz` is the
readiness probe and accounts for model loading, so an orchestrator should route
traffic only once it returns `200`:

| `status` | HTTP | Meaning |
|----------|------|---------|
| `ready` | `200` | Every configured preload target has been warmed at least once (or none is configured: the default lazy behavior is ready immediately). |
| `loading` | `503` | The background warm-up is still running. |
| `error` | `503` | A warm-up failed or exceeded `preload_timeout_sec`; the affected model entry carries the error message. |

```json
{"status":"ready","models":[{"id":"qwen3.6-35b-a3b","loaded":true}]}
```

`preload_timeout_sec` is cooperative and reporting-only: a native model load is
a blocking C call that cannot be canceled, so a budget expiry is reported as
`503 error` even while the load may still be unwinding (and flips to `ready`
once it finishes).

Readiness is sticky on success: a later idle eviction does not flip a warmed
process back to `503`. A `ready` process can still pay a cold model load on the
first request after an idle eviction. A failed warm-up stays reported until that
model loads successfully — a lazy request that loads it clears the error. Both
probes are unauthenticated.

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

OpenAI **Responses** and Anthropic **Messages** clients work against the same
base URL (`/v1`):

```bash
# OpenAI Responses API
curl -s https://api.tinycoder.dev/v1/responses \
  -H "Authorization: Bearer $TINYCODER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","input":"Write a Go hello world."}' | jq

# Anthropic Messages API
curl -s https://api.tinycoder.dev/v1/messages \
  -H "x-api-key: $TINYCODER_API_KEY" \
  -H 'anthropic-version: 2023-06-01' \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","max_tokens":256,"messages":[{"role":"user","content":"hi"}]}' | jq
```

Both endpoints are stateless translations onto the same engine: `input` /
`messages` and `instructions` / `system` are rendered into the model's chat
template, and the generated text comes back as `output_text` content / a `text`
content block. Fields the gateway does not model (`store`,
`previous_response_id`, `tools`, `metadata`, ...) are accepted and ignored;
tool calling is available on `/v1/chat/completions`. `max_tokens` is required on
`/v1/messages`, as in the Anthropic API.

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
   ([`handlers.go`](internal/server/handlers.go:103)).
2. [`Manager.Generate`](internal/server/manager.go:296) acquires a global
   concurrency slot, then a per-model lease (loading the model if needed and
   evicting an idle LRU victim when at capacity).
3. The lease's [`Engine.Generate`](internal/engine/engine.go:143) runs the
   model; tokens stream through a callback.
4. For the native engine the callback crosses into C++ via a `cgo.Handle` and
   the shim's `tc_token_cb`; the token text is copied back to Go.
5. Streaming responses write SSE chunks and flush; non-streaming responses
   aggregate the text and emit a single JSON object with usage.

### Concurrency model

- One `engine.Engine` instance per registry model, and one CPU thread pool per
  loaded model inside the engine (sized by `n_threads`, overridable per model).
- `entry.mu` serializes load/unload/generate for a model — required because the
  C++ engine owns a single KV cache and clears it per request.
- `Manager.sem` bounds admission: at most `max_concurrent_requests` requests are
  in flight, and the default is one slot per registered model (`0` in the
  config resolves to that).
- `Manager.exec` serializes the engine call itself across *all* models: the
  native engine shares one process-wide thread pool and one GPU, so two resident
  models must not generate at once. Increasing `max_concurrent_requests` changes
  how many requests wait, not how many run.
- The idle loop unloads models idle longer than `idle_timeout_sec`.
- With `max_loaded_models = N`, the manager guarantees at most `N` resident
  models by evicting the least-recently-used idle one when loading another.
  Models beyond `max_loaded_models` cannot stay warm: preloading more targets
  than that re-evicts the earlier ones.

## Configuration

Precedence: **flags > environment > JSON file > built-in defaults**.

```bash
./build/bin/tinycoder-openai \
  --config /etc/tinycoder/config.json \
  --listen :8080 \
  --models-dir /data/models
```

| Flag | Env | Default | Description |
|------|-----|---------|-------------|
| `--config` | `TINYCODER_CONFIG` | — | JSON config path |
| `--listen` | `TINYCODER_LISTEN` | `:8080` | Listen address |
| `--models-dir` | `TINYCODER_MODELS_DIR` | `/data/models` | GGUF root |
| `--log-level` | `TINYCODER_LOG_LEVEL` | `info` | `debug`/`info`/`warn`/`error` |
| `--model` | — | — | Serve exactly this model, replacing the config `models` registry |
| `--model-id` | — | slugified base name of the file without a `.gguf` extension | Served id for `--model` |
| `--ctx` | — | `0` (inherits `max_seq_len`) | Context window for the flag-provided model; `0` inherits `max_seq_len` (8192 by default) |

For one-model-per-process operation (like `llama-server`) the binary can serve
a single model without a config-file registry:

```bash
./build/bin/tinycoder-openai \
  --model /data/models/gemma/gemma4-coding-Q2_K.gguf \
  --model-id gemma4-coding --ctx 4096
```

`--model` replaces the config file's `models` registry (the flags win over the
file), so exactly that one model is served and no field is inherited from the
file's first entry. `--model-id` defaults to the slugified base name of the file
without a `.gguf` extension (other dotted suffixes are part of the name, e.g. an
EXL3 checkpoint directory), a relative `--model` is resolved against the process
working directory (not `models_dir`), and `--ctx 0` (the default) inherits
`max_seq_len` (8192 by default). `--model-id` or `--ctx` without `--model` is a
startup error.

Because `--model` replaces the registry, any `preload_models` from the config
file or `TINYCODER_PRELOAD_MODELS` must name the flag-provided id or be cleared;
otherwise startup fails with `preload_models: unknown model id`. `preload: true`
follows the single flag-provided model.

Other environment variables: `TINYCODER_MAX_LOADED_MODELS`,
`TINYCODER_IDLE_TIMEOUT_SEC`, `TINYCODER_THREADS`, `TINYCODER_MAX_SEQ_LEN`,
`TINYCODER_DEFAULT_MAX_TOKENS`, `TINYCODER_MAX_PROMPT_TOKENS`,
`TINYCODER_REQUEST_TIMEOUT_SEC`, `TINYCODER_NO_TOKEN_TIMEOUT_SEC`,
`TINYCODER_HARD_TIMEOUT_SEC`, `TINYCODER_MAX_CONCURRENT`,
`TINYCODER_ALLOWED_ORIGINS` (comma-separated), `TINYCODER_API_KEYS`
(comma-separated), `TINYCODER_TOOL_CALLING`, `TINYCODER_SCAN_MODELS_DIR`,
`TINYCODER_PRELOAD`, `TINYCODER_PRELOAD_MODELS` (comma-separated),
`TINYCODER_PRELOAD_TIMEOUT_SEC`.

`n_threads` (`TINYCODER_THREADS`) sizes each model's CPU worker pool — the
engine gives every model its own, so `n_threads` on a `models[]` entry overrides
the global value for that model (useful for running a small model on a few
threads beside a large one). `0` uses every logical CPU. `no_token_timeout_sec`
aborts a generation when neither a token nor an engine prefill-progress event
arrives for that long (a hung-engine guard; reported to the client as `504` with
code `no_token`). It defaults to `0` (disabled), because a full-GPU prefill emits
no gateway-observable progress at all, so no fixed window can distinguish a
multi-minute prefill from a hung engine; set it explicitly to get the fast-fail
back. `hard_timeout_sec` bounds the total wall-clock time of one generation
(reported as `504` with code `timeout`).

Aborts are cooperative and safe: on timeout, client disconnect or the no-token
guard, the server signals the engine's per-layer / per-block abort predicate and
then **waits for the engine call to unwind** before releasing the model lock and
concurrency slot. The loaded model (and its KV cache, which the engine clears on
abort) is reused for the next request — the model is not unloaded, and the
process cannot crash by racing a still-running engine.

### Preload and readiness

`preload` (bool, default `false`) warms the model at startup instead of on the
first request, so an orchestrator can gate traffic on `/readyz`:

- With a single entry in `"models"`, `"preload": true` is enough.
- With several models, list the ids to warm in `"preload_models"` (it takes
  precedence over `preload` when both are set). `preload: true` with several
  models and no list is a startup validation error.
- `preload_timeout_sec` (default `300`; `0` disables the bound) bounds one
  model's warm-up by canceling its load context. The budget is reporting-only:
  the context is canceled, but the engine may ignore that (a blocking native
  load), so an expiry is reported as `503 error` even while the load may still
  be running. A failed load (for example a missing model file) reports
  `503 error` on `/readyz` instead of remaining `loading`.
- Targets are warmed sequentially in the background; startup and `/healthz` are
  not delayed.

The environment equivalents are `TINYCODER_PRELOAD`,
`TINYCODER_PRELOAD_MODELS` (comma-separated) and
`TINYCODER_PRELOAD_TIMEOUT_SEC`.

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

Authentication is enforced whenever at least one key is configured. Keys are
listed in the `api_keys` array:

```json
{ "api_keys": ["sk-live-abc123", "sk-live-def456"] }
```

Clients present a configured key using any of the common OpenAI-compatible
forms; the first non-empty match wins:

| Form | Example | Used by |
|------|---------|---------|
| `Authorization: Bearer <key>` | `Bearer sk-live-abc123` | OpenAI SDKs, most clients |
| `Authorization: <key>` | `sk-live-abc123` | clients that omit the scheme |
| `api-key: <key>` | `api-key: sk-live-abc123` | Azure OpenAI, the OpenAPI `apiKey` scheme |
| `x-api-key: <key>` | `x-api-key: sk-live-abc123` | Anthropic-style compatible clients, AI SDK |
| `?api_key=<key>` | `/v1/models?api_key=sk-live-abc123` | query parameter / browser use |

A missing or unknown key receives `401` with
`{"error":{"type":"authentication_error","code":"invalid_api_key"}}`. When no
keys are configured the endpoints are open (intended only for local use);
`/healthz` and `/readyz` are always unauthenticated for probes.

See [`config.example.json`](config.example.json) for the full schema and the
editable model registry. Registry entries map a public `id` to a `file`
(relative to `models_dir`, or absolute) that is either a GGUF file or an EXL3
checkpoint directory (HF `safetensors` + `config.json` + `tokenizer.json`, e.g.
`qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw`); the engine auto-detects the format.

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

## Security

- Set `api_keys` (or `TINYCODER_API_KEYS`) in any public deployment; the server
  then requires a configured key via `Authorization: Bearer <key>`, an
  `api-key` / `x-api-key` header, or `?api_key=<key>` on all content endpoints.
  Any configured key is accepted; unknown or missing keys get `401`.
- `allowed_origins` defaults to `*`; restrict it to your front-end origins if
  you expose the API to browsers.
- `max_prompt_tokens` caps `max_tokens`: a request asking for more is clamped to
  this ceiling (not rejected), so OpenAI-compatible clients that request large
  output budgets — e.g. Roo Code, which clamps to 20% of the model's declared
  context window — still succeed. `request_timeout_sec` bounds request handling,
  the prompt-window preflight and generation, and cancels generation on client
  disconnect. Cancellation is cooperative: a native model load is a blocking C
  call that does not honor it. `no_token_timeout_sec` (default `0`, disabled)
  guards against a hung engine: it aborts when neither a token nor a
  prefill-progress event arrives within the window, so a long prefill is not
  mistaken for a hang. `hard_timeout_sec` bounds one generation call and its
  abort is cooperative: an engine that stops polling the abort predicate cannot
  be abandoned. Even with the no-token guard disabled, `request_timeout_sec`
  bounds request handling and generation; the model load itself remains
  cooperative and is not cancellable.
- Cancellation is cooperative across the C ABI: an abort predicate is polled
  inside the engine's per-layer / per-block loop, not only at the token
  callback, so a stuck prefill can be interrupted. Timeouts and cancellation are
  reported to the client as HTTP `504`, never as a truncated `200`.
