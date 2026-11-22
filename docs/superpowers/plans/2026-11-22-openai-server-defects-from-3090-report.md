# tinycoder-openai: server defects from the 2026-11-20 RTX 3090 recheck — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the three defects the rented 3090 report attributes to the server, not the engine: a 120 s no-token watchdog that kills long prefills, a `models` list from config.json that keeps fields of the built-in first model, and a prompt longer than the window returning an empty `finish_reason: "stop"` instead of HTTP 400.

**Architecture:** All three are small, local changes in the Go gateway: the watchdog must be driven by prefill progress rather than by "no token yet", `Load` must replace the models array instead of overlaying it, and the streaming handler must translate the engine's refusal into a 400 before it opens the stream.

**Tech Stack:** Go (`tinycoder-openai`, module at `/home/mike/git/tinycoder-openai`), cgo bridge to `libtinycoder_bridge.so`, `internal/config`, `internal/server`.

**Spec:** the 2026-11-20 RTX 3090 recheck report from the eva-brain stand
(received 2026-11-22; **not published in this repository**) — defects 4, 5, 6.
The engine side is `2026-11-22-engine-defects-from-3090-report.md` in the
engine repository.

## Global Constraints

- Repository: `/home/mike/git/tinycoder-openai` (commit `2a228ec` is what the report tested). The engine plan's fixes land in the other repository; do not vendor them here.
- Every task ends with `go build ./... && go test ./...` passing, and — for anything that changes what a client sees — a recorded `curl` against a locally served model.
- Config compatibility: an existing `config.json` must keep working; a *missing* `models` key still yields the built-in list.
- No silent behaviour change for OpenAI clients: keep `finish_reason` values as they are and add the error path only for prompts the engine refuses.

## Review Focus

- **A prefill that produces no tokens for 40 minutes** must not be killed while the engine is making progress; a *hung* engine must still be killed.
- **A config with `models: []`** (explicit empty) must serve no models, not the defaults.
- **A prompt exactly at the window size** must still be accepted; only `> window` is a 400.
- **`context_length_exceeded` body** must match OpenAI's shape closely enough that SDKs surface `error.code`.

---

### Task 1: The no-token watchdog must follow prefill progress

Report defect 4: on Ornith a 16k prompt takes 120 s, and the server aborts with `"stream generation ended" err="no token produced within timeout"` while still sending keep-alives. Engine log lines exist for prefill progress (`[TinyCoder] Prefill: …` and per-step timing), but the gateway only sees "no token".

**Files:**
- Modify: `internal/config/config.go:50` (`NoTokenTimeoutSec`) — document `0 = disabled`
- Modify: `internal/server/*.go` — the watchdog that emits "no token produced within timeout"
- Test: `internal/server/*_test.go` (new)

**Interfaces:**
- Produces: `func (s *Server) prefillProgress() time.Time` or an equivalent channel; the watchdog resets on either a token or a progress event.
- Consumes: the bridge's progress callback if it exists; otherwise the engine's stderr line reader, which already exists for logs.

- [ ] **Step 1: Reproduce** — serve Ornith Q4_K_M with `no_token_timeout_sec: 120`, send a 16k-token prompt, observe the abort while `[TinyCoder] Prefill:` keeps advancing. Save the log.
- [ ] **Step 2: Write the failing test** — a fake engine that emits progress every 10 s and no token for 300 s, with `NoTokenTimeoutSec: 60`; assert the request survives. Today it fails.
- [ ] **Step 3: Implement** — reset the watchdog on prefill progress as well as on tokens; keep a separate hard ceiling (`NoTokenTimeoutSec` measured from the last progress event). `0` disables both.
- [ ] **Step 4: Test both directions** — progress ⇒ survives; no progress at all ⇒ still aborted with the same message.
- [ ] **Step 5: Document the default** — change the default to `0` (disabled) or keep the value but explain in `README` that it counts from the last *progress* event; pick one and make the config comment say it.
- [ ] **Step 6: Commit** — `git commit -m "server: watchdog follows prefill progress, not first token"`.

### Task 2: A config `models` list replaces the built-in one

Report defect 5: a one-model config yields `/v1/models` entries carrying `"architecture":"qwen35moe","quantization":"UD-Q4_K_M","description":"35B-A3B mixture-of-experts coding model (recommended)."` from the built-in first model. `Load` (`internal/config/config.go:148-178`) merges JSON key-wise, which *should* replace the array; the leak therefore comes from somewhere else — find it before changing anything.

**Files:**
- Modify: `internal/config/config.go` (`Load` and/or `Default`) once the merge site is known
- Modify: `internal/server/models.go` (or wherever `/v1/models` is assembled)
- Test: `internal/config/config_test.go`

**Interfaces:**
- Produces: `Load` guarantees that a `models` key present in the file yields exactly those models, with zero-value fields for anything the file omits.
- Consumes: nothing new.

- [ ] **Step 1: Reproduce** — write a config with one model and no `architecture`/`quantization`/`description`; `curl /v1/models` and record the leaked fields.
- [ ] **Step 2: Find the merge** — log the slice length before and after `Load`, and grep for where the served registry is built (`grep -rn "v1/models" internal`). Write down which of the two merges leaks.
- [ ] **Step 3: Write the failing test** — `Load` on that file returns `len(cfg.Models) == 1` with empty `Architecture`, and the served list has one entry.
- [ ] **Step 4: Fix the merge** — if `Load` is the site, replace the slice instead of overlaying; if the registry is, build it from `cfg.Models` alone.
- [ ] **Step 5: Verify** — the curl from Step 1 shows one clean entry; a config *without* `models` still serves the defaults.
- [ ] **Step 6: Commit** — `git commit -m "server: config models list replaces the built-in registry"`.

### Task 3: A prompt longer than the window returns HTTP 400 `context_length_exceeded`

Report defect 6: the engine logs `generate: prompt refused (window 33536 tokens, prompt 115204)` and the server returns a stream with `finish_reason: "stop"` and zero tokens. The window is known to the server from `/v1/models`.

**Files:**
- Modify: `internal/server/chat.go` (or the handler that opens the stream)
- Test: `internal/server/chat_test.go`

**Interfaces:**
- Consumes: the model's window from the registry entry.
- Produces: for `promptTokens > window`, a 400 before any streaming headers, body `{"error":{"message":"...","type":"invalid_request_error","code":"context_length_exceeded"}}`.

- [ ] **Step 1: Reproduce** — request a prompt longer than the served window; capture the empty stream.
- [ ] **Step 2: Write the failing test** — handler test with a fake model of window 1024 and a 2000-token prompt; assert status 400, `error.code == "context_length_exceeded"`, and no stream opened.
- [ ] **Step 3: Implement** — tokenize (the bridge already exposes `tc_engine_tokenize`), compare against the window, and return the error *before* the stream is created. Keep the engine's refusal as a fallback: if the engine still refuses mid-request, map it to the same body instead of an empty success.
- [ ] **Step 4: Verify** — same request now 400s; a prompt of exactly `window` tokens still streams.
- [ ] **Step 5: Commit** — `git commit -m "server: 400 context_length_exceeded instead of an empty stream"`.

## Not in this plan

- **`/healthz` before the model is loaded, and a `preload` option** (report §4.5) — a readiness change wants its own design (liveness vs readiness split, warm-up budget).
- **CLI flags `--model/--model-id/--ctx`** (§4.6) — additive surface, separate change.
- **Per-token overhead 7–44 %** (§4.7) — needs a CPU profile of the gateway against a fast model; measure before changing anything.
