# tinycoder-openai: remaining 2026-11-20 RTX 3090 report items — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the three report items the defect plan left out: readiness that accounts for model loading plus a config `preload`; `--model`/`--model-id`/`--ctx` flags for one-model-per-process operation; and a measured answer to the gateway's 7–44 % per-token generation loss — measure first, change only what the profile justifies.

**Architecture:** Readiness is a server-level `/readyz` (liveness `/healthz` stays untouched) backed by a background preloader that warms configured models through the existing `Manager.Acquire`/`Release`; the CLI flags are a config-level override applied after file/env resolution, so precedence stays flags > env > file > defaults; the overhead work is a benchmark harness that separates engine time, gateway/SSE time and cgo bridging time, plus a short report. No engine-repository changes.

**Tech Stack:** Go 1.22, cgo bridge (`libtinycoder_bridge.so`), `internal/config`, `internal/server`, `cmd/tinycoder-openai`, `internal/engine`.

**Spec:** the 2026-11-20 RTX 3090 recheck report from the eva-brain stand (§4.5–§4.7), received 2026-11-22; the report body is not published in this repository, its relevant text is quoted per task below.

## Global Constraints

- Repository: `/home/mike/git/tinycoder-openai`, branch `fix/server-defects-2026-11-22` (three defect tasks already committed). Do not change those defect semantics.
- Additive surface only: an existing `config.json` and the existing HTTP/CLI behavior must keep working when the new options are absent.
- Every task ends with `go build ./... && go test ./...` passing, and — for anything a client or operator can observe — a recorded `curl`/CLI transcript.
- Do not modify the engine sources under `build/native/_deps/`; the engine plans land in the other repository.
- `docs/` is tracked (the plans live there); `.superpowers/` is scratch and must never be committed.

## Review Focus

- `/healthz` must stay a pure liveness probe (200 while the process serves), and `/readyz` must be 503 until the configured warm-up finished, then 200 even if the idle loop later evicts the model (readiness is "warmable", not "resident forever").
- `--model` must mean the same thing a relative config `file` means: a path usable as given from the shell, not relative to `models_dir`.
- The overhead measurement must compare like with like (same model, same prompt, same sampling parameters, warm model, no client-side bottleneck); a number without that control is not evidence.

---

### Task F1: Readiness and preload

Report §4.5 (quoted): `/healthz` answers 200 before the model is loaded (it is loaded lazily on the first request); the reporter wants a `preload` in the config and a readiness signal that accounts for loading — "we warm it up ourselves now".

**Files:**
- Modify: `internal/config/config.go` (+ `internal/config/config_test.go`)
- Modify: `internal/server/manager.go`, `internal/server/server.go`
- New: `internal/server/readiness.go`, `internal/server/readiness_test.go`
- Modify: `cmd/tinycoder-openai/main.go`, `README.md`, `config.example.json`

**Interfaces (exact):**
- `Config` gains:
  ```go
  // Preload warms the registry at startup instead of on the first request.
  // With several registered models it requires preload_models.
  Preload bool `json:"preload"`
  // PreloadModels lists the model ids to warm at startup (takes precedence
  // over Preload when both are set).
  PreloadModels []string `json:"preload_models"`
  // PreloadTimeoutSec bounds one model's warm-up (0 disables the bound).
  PreloadTimeoutSec int `json:"preload_timeout_sec"`
  ```
  Defaults: `Preload:false`, `PreloadModels:nil`, `PreloadTimeoutSec:300`.
- `func (c Config) PreloadIDs() []string` — `PreloadModels` when non-empty; else, when `Preload` is true, the single registry model's id; else nil.
- `Validate()` additions: every `preload_models` id must exist in `Models` (error `preload_models: unknown model id %q`); `Preload` true with neither `preload_models` nor exactly one model is an error (`preload requires a single registered model or preload_models`); `PreloadTimeoutSec < 0` is an error.
- `envInt("TINYCODER_PRELOAD_TIMEOUT_SEC", &c.PreloadTimeoutSec)` and `envBool("TINYCODER_PRELOAD", &c.Preload)` in `ApplyEnv`; `TINYCODER_PRELOAD_MODELS` (CSV → `PreloadModels`) too.
- `Server` gains, in `internal/server/readiness.go`:
  ```go
  // StartPreload warms the configured models in the background. It returns
  // immediately; readiness reports progress. No-op when no model is configured.
  func (s *Server) StartPreload()
  ```
  and `GET /readyz` routed in `server.go` (no auth, like `/healthz`).
- Readiness response (`Content-Type: application/json`):
  - 200 `{"status":"ready","models":[{"id":"m","loaded":true}]}` once every preload target has been warmed at least once (or when there are no targets — lazy behavior, ready immediately);
  - 503 `{"status":"loading","models":[{"id":"m","loaded":false}]}` while the background warm-up runs;
  - 503 `{"status":"error","models":[{"id":"m","loaded":false,"error":"..."}]}` when a warm-up failed or timed out.
  Readiness is sticky on success: an idle eviction later must not flip it back to 503 (the model stays warmable; the process is healthy). It flips to 200 if a later request or retry loads the failed model.
- Warm-up mechanics: `Acquire(ctx, id)` then `Release()` per target, sequentially (loading several models concurrently would thrash `max_loaded_models`), each bounded by `PreloadTimeoutSec` when non-zero; failures are logged at error level with the model id and stored for `/readyz`; a failed target is retried on the next `/readyz`? No — no retry loop; `/readyz` reports it, and the lazy path plus a subsequent successful load flips it to ready (record success whenever `Acquire` succeeds).
  Implementation note: implement the warm-up state inside `Manager` (`func (m *Manager) Preload(ctx context.Context, ids []string)`, `func (m *Manager) Readiness(ids []string) (status string, models []ModelReadiness)`) so the state is shared with lazy loads; `Server.StartPreload` is the thin launcher. `ModelReadiness` is `struct { ID string; Loaded bool; Error string }`.
- `main.go`: after the manager and handler are built (and before `ListenAndServe`), `srv.StartPreload()`; log `preloading models` with the ids. Startup must not block on the warm-up.

**Tests:**
- `internal/config/config_test.go`: `PreloadIDs` (explicit list wins; `preload:true` + one model; `preload:true` + two models + no list ⇒ `Validate` error; unknown id ⇒ `Validate` error); `PreloadTimeoutSec` default 300 and env override; an existing config without the new keys still validates.
- `internal/server/readiness_test.go`: no targets ⇒ `/readyz` 200 `ready` immediately and `/healthz` 200; a fake engine whose `Load` blocks on a channel ⇒ `/readyz` 503 `loading` while blocked while `/healthz` stays 200, then 200 `ready` after release; a fake whose `Load` returns an error ⇒ `/readyz` 503 `error` whose model entry carries the message, `/healthz` 200; a warm-up budget test with `PreloadTimeoutSec:1` and a `Load` that blocks ⇒ 503 `error` mentioning the timeout after ≈1 s; after a successful warm-up, `Manager` unload (via an idle-eviction-equivalent call or `mgr` entry helper) ⇒ `/readyz` stays 200.
- Everything else must keep passing unchanged.

**Steps:**
- [ ] Step 1: Write the failing config tests and the failing readiness tests (RED), capturing output.
- [ ] Step 2: Implement config fields, `PreloadIDs`, `Validate`, env; then manager/Serve readiness; route `/readyz`; start the preloader from `main`.
- [ ] Step 3: GREEN; then record the curl transcript: start the stub binary with a `preload:true` config whose model file does not exist (so warm-up fails fast) and show `/healthz` 200 vs `/readyz` 503 `error` with the engine message; then start it with a config whose warm-up succeeds (stub engine loads any path) and show `/readyz` 200. Save both transcripts.
- [ ] Step 4: Document `preload`, `preload_models`, `preload_timeout_sec` and `/readyz` in `README.md` (configuration table/paragraphs and an operational-endpoints note) and in `config.example.json`; note that `/healthz` remains liveness-only.
- [ ] Step 5: `gofmt -l`, `go vet ./...`, `go build ./...`, `go test ./...`; commit `server: readiness endpoint and config preload`.

### Task F2: `--model`, `--model-id`, `--ctx` flags

Report §4.6 (quoted): several parameters of one model can only be set through `config.json`; for one-model-per-process operation (like `llama-server`) the binary itself should accept `--model`, `--model-id`, `--ctx`.

**Files:**
- Modify: `internal/config/config.go` (+ `internal/config/config_test.go`)
- Modify: `cmd/tinycoder-openai/main.go`, `README.md`

**Interfaces (exact):**
- `func (c *Config) ApplyModelFlags(model, modelID string, ctx int) error`:
  - `model == ""`: return an error when `modelID != ""` or `ctx != 0` (`--model-id/--ctx require --model`); otherwise nil.
  - Resolve `model` with `filepath.Abs` (a relative CLI path is relative to the shell's working directory, not `models_dir`; `Config.ResolveModelPath` returns absolute paths as-is).
  - `ctx < 0` ⇒ error `--ctx must be >= 0`.
  - `modelID == ""` ⇒ derive it from the file base name with the existing `slugify` (e.g. `/x/Ternary-Bonsai-2-27B-PQ2_0.gguf` → `ternary-bonsai-2-27b-pq2-0`).
  - Replace the registry: `c.Models = []RegistryModel{{ID: modelID, Name: baseNameWithoutExt, File: absolutePath, ContextLength: ctx}}`. Anything the flags omit stays zero (no field inherited from the file's first model).
- `main.go`: flags `--model`, `--model-id`, `--ctx` (int, default 0), applied after `config.Load` + `ApplyEnv` and before `Validate` (flags > env > file > defaults); `--ctx` is `flag.Int`.
- Tests: `internal/config/config_test.go` — derived id, explicit id, `ctx` forwarded, relative path made absolute, errors for `--model-id`/`--ctx` without `--model` and negative ctx, and override of a config file's `models` list (one entry, no leaked fields).
- README: add the three flags to the flags table and a one-model-per-process example:
  ```bash
  ./build/bin/tinycoder-openai --model /data/models/gemma/gemma4-coding-Q2_K.gguf --model-id gemma4-coding --ctx 4096
  ```
  plus a sentence that `--model` replaces the config registry (it wins over the file).

**Steps:**
- [ ] Step 1: failing config tests (RED), captured.
- [ ] Step 2: implement `ApplyModelFlags` + main flags; GREEN.
- [ ] Step 3: record the CLI transcript against the stub binary: `--model` pointing at a scratch `.gguf` that does not exist + `--model-id solo --ctx 4096` ⇒ `/v1/models` shows exactly `solo` with `context_length:4096` (the stub loads any path, so no real GGUF needed); and `--model-id` without `--model` ⇒ startup fails with the documented error.
- [ ] Step 4: README updates; `go build/test ./...`; commit `cli: --model/--model-id/--ctx for one-model processes`.

### Task F3: Measure, then reduce, the per-token gateway overhead

Report §4.7 (quoted): the server loses 7–44 % of generation relative to `tinycoder_bench` (5.5) — Qwen2.5-Coder-1.5B 416 → 234 tok/s, Qwen3.6-27B 33 → 20, Bonsai 69 → 57. The faster the model the larger the loss, which points at a fixed per-token cost in the gateway/bridge (detokenization, SSE, cgo).

**Files:**
- New: `internal/server/overhead_bench_test.go`
- New: `docs/superpowers/reports/2026-11-22-gateway-token-overhead.md`
- Modify (only if the measurement justifies it): `internal/server/handlers.go` (SSE write path)

**Method (binding — measure before changing anything):**
- Benchmarks skip unless `TINYCODER_BENCH_MODEL` names a real GGUF on this host:
  - `BenchmarkEngineDirect`: `native` engine loaded once, `Generate` with the gateway's default sampling parameters (`Temperature .7, TopP .9, TopK 40, RepeatPenalty 1.1, RepeatLastN 64`) and a fixed short prompt, `MaxTokens=128`, token callback that only counts. Reports ns/op and tokens/s.
  - `BenchmarkGatewayStream`: the same model/prompt/params/max_tokens through `httptest` + `POST /v1/chat/completions` with `"stream":true`, consuming the SSE body to `[DONE]`. Reports ns/op and tokens/s.
  - `BenchmarkSSEWrite`: the token-write path alone (`sseWriter.write` of a chat chunk, fixed text, a discard `ResponseWriter` with `Flush`), reporting the pure gateway per-token cost with no engine.
- Controls that must be stated in the report: model loaded once and kept warm before timing; the same prompt and sampling parameters; the client consumption loop must not throttle (read all bytes, no per-token processing); tokens generated counted from the SSE deltas for both benchmarks; the benchmark run on this host with the command printed.
- Derive: `engine_ns_per_token`, `gateway_ns_per_token`, overhead = `(gateway - engine) / engine`; also report `SSEWrite` share. Run `-cpuprofile` and `go tool pprof -top` and paste the top entries.
- Decision rule: if `SSEWrite` + per-token gateway work is under ~2 % of engine time at the fastest measured model, the 7–44 % figure is not gateway-side (look at parameter/engine-path differences and say so in the report — do not change code). If it is larger, implement the smallest optimization the profile justifies and re-measure before/after with the same command.
- Report file contents: the commands, host/model, the raw numbers, the decomposition, the profile top, the before/after (if any), and the conclusion — including an explicit statement that the original 416→234 tok/s comparison is to a different harness (`tinycoder_bench`) and which factors were controlled here.

**Steps:**
- [ ] Step 1: Write the benchmark harness; run it with a locally available model (`/data/models/gemma/gemma4-coding-Q2_K.gguf` and/or `/data/models/bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf`; never download anything). Capture the numbers.
- [ ] Step 2: Profile (`-cpuprofile`) and record the top functions.
- [ ] Step 3: Write the report with the decomposition and the decision.
- [ ] Step 4: If the profile justifies a fix, implement it under test (existing tests must pass, SSE frames unchanged byte-for-byte for identical inputs) and re-measure.
- [ ] Step 5: `go build/test ./...`; commit `bench: gateway per-token overhead (measured)` — the benchmark and report are committed even when no code change follows.

## Not in this plan

- Continuous batching, GPU backend changes, or anything in the engine repository.
- Changing `max_prompt_tokens`/`max_tokens` policy, auth, or the model registry semantics fixed by the defect plan.
