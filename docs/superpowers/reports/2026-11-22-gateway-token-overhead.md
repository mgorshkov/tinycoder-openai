# Gateway per-token overhead: measurement (2026-11-22)

Task F3 of the server follow-up plan. Question: does the OpenAI-compatible HTTP
gateway add a fixed per-token cost that explains the report's 7–44 % generation
loss relative to `tinycoder_bench` (§4.7: Qwen2.5-Coder-1.5B 416 → 234 tok/s,
Qwen3.6-27B 33 → 20, Bonsai 69 → 57)?

**Answer: no. On this host the pure gateway/SSE serialization cost is ~2.6 µs per
token (0.011–0.017 % of engine per-token time), and the whole end-to-end HTTP
path — server handler, socket transport and the client's SSE consumer — adds
0.36 % (Bonsai) to 0.94 % (faster 9B model) per token. The 7–44 % figure does not
reproduce as gateway overhead here. No production code was changed.**

## Host, models, harness

| | |
|---|---|
| CPU | Intel Xeon E5-2686 v4, 72 logical CPUs |
| GPU | NVIDIA RTX 2080 Ti, 11 GB, driver 595.91.07 |
| Go | go1.26.8 linux/amd64, CGO on, native bridge `build/native/libtinycoder_bridge.so` |
| Repo | `fix/server-defects-2026-11-22` at `c4f83a8` (F1/F2 fixes included) |
| Engine | tinycoder-inference through `internal/engine/native`; 32 layers offloaded, 32 on CPU, engine thread pool 72 threads |

Models (loaded once per benchmark process; never downloaded):

| id | file | size | engine-reported load |
|---|---|---|---|
| Bonsai | `/data/models/bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5.95 GB | weights 59.8 s + embedding pre-dequant 13.3 s |
| Ornith 9B | `/data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf` | 5.91 GB | weights 0.4 s + embedding pre-dequant 14.3 s |

`/data/models/gemma/gemma4-coding-Q2_K.gguf` (the planned faster second model)
**fails to load** in this engine build:

```
[TinyCoder] Missing attention weights for layer 5
[TinyCoder] Failed to load model weights
load: ... Failed to load model weights
```

so the faster second measurement uses the other locally available, smaller
model (Ornith 9B), which loads and runs at ~64 tok/s vs Bonsai's ~43 tok/s.

Harness: `internal/server/overhead_bench_test.go` (new, package `server`). It
skips unless `TINYCODER_BENCH_MODEL` names a readable GGUF and is always
skipped in a `tinycoder_stub` build. The model is loaded once via
`wiring.NewFactory()("bench")` + `Load({ModelPath, MaxSeqLen: 4096})`; the
`Manager` receives a wrapper whose `Load` is a no-op and whose `Loaded` is true,
so `Manager.Acquire` takes its resident fast path and never reloads.

Commands (exact):

```sh
# engine-direct vs end-to-end HTTP streaming, 10 iterations x 5 counts each
TINYCODER_BENCH_MODEL=/data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf \
  go test ./internal/server -run '^$' \
  -bench 'Benchmark(EngineDirect|GatewayStream)$' -benchtime=10x -count=5 -timeout 60m

TINYCODER_BENCH_MODEL=/data/models/bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  go test ./internal/server -run '^$' \
  -bench 'Benchmark(EngineDirect|GatewayStream)$' -benchtime=10x -count=3 -timeout 60m

# paired cross-check: direct and gateway back-to-back on the same seed, order
# alternated to cancel GPU-clock drift between the two separate bursts
TINYCODER_BENCH_MODEL=/data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf \
  go test ./internal/server -run '^$' -bench 'BenchmarkGatewayDelta$' -benchtime=20x
TINYCODER_BENCH_MODEL=/data/models/bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  go test ./internal/server -run '^$' -bench 'BenchmarkGatewayDelta$' -benchtime=20x

# pure SSE serialization, time-based so ns/op is not fixed-cost inflated by a
# 10-iteration run
TINYCODER_BENCH_MODEL=/data/models/bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  go test ./internal/server -run '^$' -bench 'BenchmarkSSEWrite$' -benchtime=2s -count=3
```

## Controls held on both paths

- Same model instance, loaded once before timing; warmed with one direct
  `Generate` and one full gateway streaming request before `b.ResetTimer()`.
- Same request: chat `Messages` (so the chat template render is inside both
  numbers; 57 prompt tokens for this prompt on both models), `MaxTokens=128`,
  `Temperature 0.7, TopP 0.9, TopK 40, RepeatPenalty 1.1, RepeatLastN 64`
  (the gateway defaults). The HTTP body sets only `model`, `messages`,
  `max_tokens`, `stream`, `seed`; every other value is the gateway default.
- Both callbacks go through the same cgo shim (`tcGoTokenCallback`), so the
  bridge's per-token transition and detokenization are inside BOTH numbers.
  The delta isolates the gateway layer, not the bridge.
- Sampling at temperature 0.7 is stochastic and the engine keeps a persistent
  RNG (seed 0 seeds once from `random_device`, then continues), so the two
  paths cannot otherwise be guaranteed to emit the same number of tokens. Each
  iteration pins the same seed (cycling 1..8) in both requests; a changing seed
  forces the sampler to reseed, making the token sequence a function of
  (prompt, seed). The per-seed token count is measured once in setup and
  asserted on both paths; every measured iteration produced exactly 128
  completion tokens.
- HTTP client consumes every byte up to `[DONE]`, counting non-empty
  `delta.content` values; it does no other per-token work. Direct callback only
  counts.
- `NoTokenTimeoutSec: 0`, `MaxConcurrent: 1`, `MaxLoadedModels: 1`, discard
  `slog` logger (keeps access-log formatting cost but not I/O).
- The end-to-end number necessarily includes the client's read/parse and the
  loopback TCP transport, because that is what "the HTTP path" costs a consumer.
  `BenchmarkSSEWrite` isolates the server-side serialization part.

## Raw results

Benchmark lines (`-benchtime=10x`; `ns/op` is per 128-token request):

Ornith 9B (faster model) — first 5 lines direct, last 5 gateway:

```
      10	2100898059 ns/op	        60.93 tok/s	       128.0 tokens/op	    1896 B/op	     141 allocs/op
      10	2008533706 ns/op	        63.73 tok/s	       128.0 tokens/op	    1846 B/op	     141 allocs/op
      10	2013199351 ns/op	        63.58 tok/s	       128.0 tokens/op	    1980 B/op	     141 allocs/op
      10	2081866343 ns/op	        61.48 tok/s	       128.0 tokens/op	    1742 B/op	     141 allocs/op
      10	2013971708 ns/op	        63.56 tok/s	       128.0 tokens/op	    1739 B/op	     141 allocs/op
      10	2068542892 ns/op	        61.88 tok/s	       128.0 tokens/op	  241763 B/op	    3078 allocs/op
      10	2085181928 ns/op	        61.39 tok/s	       128.0 tokens/op	  237012 B/op	    3075 allocs/op
      10	2076508741 ns/op	        61.64 tok/s	       128.0 tokens/op	  240443 B/op	    3080 allocs/op
      10	2089498563 ns/op	        61.26 tok/s	       128.0 tokens/op	  247549 B/op	    3080 allocs/op
      10	2087706197 ns/op	        61.31 tok/s	       128.0 tokens/op	  242990 B/op	    3074 allocs/op
```

Bonsai 27B (report's data point; `-count=3`) — first 3 direct, last 3 gateway:

```
      10	3010042590 ns/op	        42.52 tok/s	       128.0 tokens/op	    1790 B/op	     141 allocs/op
      10	3004979227 ns/op	        42.60 tok/s	       128.0 tokens/op	    1838 B/op	     141 allocs/op
      10	3000942392 ns/op	        42.65 tok/s	       128.0 tokens/op	    1691 B/op	     141 allocs/op
      10	3035545130 ns/op	        42.17 tok/s	       128.0 tokens/op	  240523 B/op	    3080 allocs/op
      10	3018796188 ns/op	        42.40 tok/s	       128.0 tokens/op	  244338 B/op	    3079 allocs/op
      10	3030366650 ns/op	        42.24 tok/s	       128.0 tokens/op	  251466 B/op	    3084 allocs/op
```

Pure SSE serialization (`sseWriter.write` of a fixed `chatChunk` through a
discard `ResponseWriter` with `Flush`, no engine):

```
BenchmarkSSEWrite-72    	 1000000	      2586 ns/op	     282 B/op	       3 allocs/op
BenchmarkSSEWrite-72    	  884095	      2651 ns/op	     282 B/op	       3 allocs/op
BenchmarkSSEWrite-72    	  858702	      2620 ns/op	     282 B/op	       3 allocs/op
```

### Derived per-token numbers

| model | path | ms/op (median) | ms/token | tok/s median (range) |
|---|---|---|---|---|
| Ornith 9B | engine direct | 2013.2 | 15.728 | 63.58 (60.93–63.73) |
| Ornith 9B | gateway stream | 2076.5 | 16.223 | 61.64 (61.26–61.88) |
| Bonsai 27B | engine direct | 3005.0 | 23.476 | 42.60 (42.52–42.65) |
| Bonsai 27B | gateway stream | 3030.4 | 23.675 | 42.24 (42.17–42.40) |

The per-path medians above are over different positions in a timed burst, so
they are not a paired comparison (Ornith's medians differ by 3.0 %, Bonsai's by
0.8 %); the per-count deltas below and the paired benchmark are the estimators.

Per-count gateway − engine deltas (adjacent runs in one process):

| model | delta ms/request | delta % | per-token delta |
|---|---|---|---|
| Ornith 9B | −32.4, +76.7, +63.3, +7.6, +73.7 | mean +1.89 %, median +3.15 % | mean 0.295 ms, median 0.495 ms |
| Bonsai 27B | +25.5, +13.8, +29.4 | mean +0.76 %, median +0.85 % | mean 0.179 ms, median 0.199 ms |

The two benchmark functions are timed as separate 20 s bursts, so a slow GPU
clock drift between the bursts shows up as run-to-run variance of the engine
itself (Ornith direct: 2008–2101 ms, ±2.3 %), and the gateway burst always runs
second within a `-count` iteration (hotter GPU). `BenchmarkGatewayDelta` is a
paired cross-check that removes this bias: one direct and one gateway request
per iteration on the same seed, with the order alternated. Its own ns/op covers
two generations plus two seed resets and is not meaningful; the reported
metrics are the per-path means and the delta:

```
# Ornith 9B, -benchtime=20x
      20	4338319773 ns/op	  19474919 delta-ns/op	      2079 direct-ms/op	         0.9366 gateway-delta-%	      2099 gateway-ms/op

# Bonsai 27B, -benchtime=20x
      20	6402709409 ns/op	  11067609 delta-ns/op	      3053 direct-ms/op	         0.3625 gateway-delta-%	      3064 gateway-ms/op
```

So the drift-cancelled end-to-end gateway delta is **+0.94 % per request on the
faster model and +0.36 % on Bonsai** (19.5 ms and 11.1 ms per 128-token
request, i.e. 0.152 ms/token and 0.087 ms/token). An earlier ad-hoc version of
the same paired check measured +1.33 % mean on Ornith, so the estimate is stable
at ~1 % and comfortably below the decision rule's ~2 % threshold.

## Decomposition (paired benchmark)

| model | engine ms/op | gateway ms/op | delta per request | delta per token | SSE share of engine token time | rest of delta |
|---|---|---|---|---|---|---|
| Ornith 9B | 2079 | 2099 | +19.5 ms | +0.152 ms | 0.0026 / 16.24 = **0.016 %** | ~0.94 % |
| Bonsai 27B | 3053 | 3064 | +11.1 ms | +0.087 ms | 0.0026 / 23.85 = **0.011 %** | ~0.35 % |

`BenchmarkSSEWrite` isolates the server serialization part; everything else in
the delta is socket transport, the client's SSE read/parse, and scheduling. An
ad-hoc in-process probe (`httptest.ResponseRecorder` — same handler, no TCP, no
HTTP client, `Flush` a no-op) was indistinguishable from the direct benchmark
(2091 ms vs 2116 ms in the same run), while a real HTTP client added ~30–40 ms
per request. In other words: the server-side handler and SSE code cost ~nothing
measurable per token; the measurable end-to-end delta is transport and client
consumption, most of which a real deployment does not pay server-side.

For reference, at a hypothetical 416 tok/s (2.4 ms/token) the measured
~0.15 ms/token delivery cost would be ~6 % — the right *shape* for the
report's "faster model loses more", but far short of 44 %, and it is not
fixed by changing the server's serialization.

## CPU profile

Gateway benchmark, Ornith, `-benchtime=20x`, via a compiled test binary
(`-test.cpuprofile`). Note the profile also covers the one-time model load
(`tc_engine_load`, 18.7 s):

```
File: f3-server.test
Build ID: c1c1cb661ad6513d3ad618cd1e2a8128f9d09753
Type: cpu
Time: 2026-11-22 18:07:56 MSK
Duration: 82.10s, Total samples = 82.21s (100.14%)
Showing nodes accounting for 81.72s, 99.40% of 82.21s total
Dropped 209 nodes (cum <= 0.41s)
      flat  flat%   sum%        cum   cum%
    81.72s 99.40% 99.40%     81.85s 99.56%  runtime.cgocall
         0     0% 99.40%     63.16s 76.83%  github.com/mgorshkov/tinycoder-openai/internal/engine/native.(*Engine).Generate.func1
         0     0% 99.40%     63.16s 76.83%  github.com/mgorshkov/tinycoder-openai/internal/engine/native.(*Engine).Generate.func1.2 (inline)
         0     0% 99.40%     18.66s 22.70%  github.com/mgorshkov/tinycoder-openai/internal/engine/native.(*Engine).Load
         0     0% 99.40%     18.66s 22.70%  github.com/mgorshkov/tinycoder-openai/internal/engine/native.(*Engine).Load.func2 (inline)
         0     0% 99.40%     63.16s 76.83%  github.com/mgorshkov/tinycoder-openai/internal/engine/native._Cfunc_tc_engine_generate
         0     0% 99.40%     18.66s 22.70%  github.com/mgorshkov/tinycoder-openai/internal/engine/native._Cfunc_tc_engine_load
         0     0% 99.40%     18.80s 22.87%  github.com/mgorshkov/tinycoder-openai/internal/server.BenchmarkGatewayStream
         0     0% 99.40%     18.67s 22.71%  github.com/mgorshkov/tinycoder-openai/internal/server.benchSetup
         0     0% 99.40%     18.67s 22.71%  github.com/mgorshkov/tinycoder-openai/internal/server.benchSetup.func1
         0     0% 99.40%     18.67s 22.71%  github.com/mgorshkov/tinycoder-openai/internal/server.newBenchEnv
         0     0% 99.40%     18.67s 22.71%  sync.(*Once).Do (inline)
         0     0% 99.40%     18.67s 22.71%  sync.(*Once).doSlow
         0     0% 99.40%     18.69s 22.73%  testing.(*B).run1.func1
         0     0% 99.40%     18.80s 22.87%  testing.(*B).runN
```

The gateway benchmark is 99.4 % `runtime.cgocall` (the engine call); no Go
gateway function (`json.Marshal`, `sseWriter.write`, `http` server, client read
loop) reaches the 0.4 s sample cutoff. There is no server-side hotspot the
profile justifies optimising.

Supplemental profile of the isolated serialization path (1M frames):

```
      flat  flat%   sum%        cum   cum%
     400ms 11.70% 11.70%     1530ms 44.74%  encoding/json.structEncoder.encode
     260ms  7.60% 19.30%      290ms  8.48%  encoding/json.appendString[go.shape.string]
     140ms  4.09% 23.39%      140ms  4.09%  runtime.(*lfstack).pop
     130ms  3.80% 27.19%      130ms  3.80%  runtime.memmove
      90ms  2.63% 29.82%       90ms  2.63%  bytes.(*Buffer).tryGrowByReslice (inline)
      80ms  2.34% 32.16%      110ms  3.22%  bytes.(*Buffer).WriteString
      80ms  2.34% 34.50%      160ms  4.68%  reflect.Value.Field
      70ms  2.05% 43.27%      510ms 14.91%  encoding/json.stringEncoder
      40ms  1.17% 57.02%      270ms  7.89%  fmt.Fprintf
```

Even the serialization cost is dominated by `encoding/json`; the total is
2.6 µs/frame, so the maximum possible saving from hand pre-serializing a frame
is well under 2 µs/token — 0.013 % of engine time at 64 tok/s. The decision
rule does not justify touching production code, and the profile does not
identify a hotspot worth changing.

## Comparison with the report (§4.7)

| report data point (3090, `tinycoder_bench` 5.5 vs server) | this measurement (2080 Ti, engine direct vs HTTP gateway) |
|---|---|
| Qwen2.5-Coder-1.5B 416 → 234 tok/s (−44 %) | not available locally |
| Qwen3.6-27B 33 → 20 tok/s (−39 %) | not available locally |
| Bonsai 69 → 57 tok/s (−17 %) | 42.6 → 42.2 tok/s (paired −0.36 %; separate-burst mean −0.76 %) |
| faster model ⇒ larger loss | 9B model: −0.9 % (paired; −1.9 % separate-burst mean); 27B Bonsai: −0.4 % (same direction, far smaller) |

Factors that differ and plausibly explain the gap; they were controlled or
excluded here:

1. **Different harness on the baseline side.** `tinycoder_bench` calls the
   engine directly; the server number is an HTTP/SSE client measurement. Any
   fixed per-token consumer/transport cost is divided by a shrinking per-token
   engine time as models get faster. Our end-to-end fixed cost is
   ~0.19 ms/token; at 416 tok/s that would be ~8 %, not 44 %, but it is the
   same mechanism and it is *client/transport*, not gateway serialization.
2. **Different request semantics.** This harness renders chat `Messages`
   through the model's chat template (57 prompt tokens) on both sides, so the
   template is not a server-side penalty here. `tinycoder_bench` may use a raw
   prompt, different sampler values, a different `max_tokens`, or a different
   measurement window (steady-state generation vs. a whole short request
   including prefill/first token).
3. **Different engine build/configuration.** This branch contains the F1/F2
   fixes; the report predates them. Thread count/offload split, engine version,
   and `n_threads` differ (this host runs 72 engine threads on a 72-vCPU box).
4. **Different host.** RTX 2080 Ti 11 GB vs the report's 3090. Absolute
   throughput is lower here (Bonsai 42.6 tok/s vs the report's 69 direct), so
   absolute numbers are not comparable; the *ratio* is what this task tests.
5. **Per-token bridge/cgo is ruled out by construction.** Both paths use the
   same `tcGoTokenCallback`/shim, so detokenization and the cgo transition are
   inside both numbers. They cannot be the gateway-specific delta.
6. **Run-to-run noise.** The engine itself varies ±2–3 % per 128-token request
   on this host (GPU clock), which is larger than the gateway delta; the
   committed separate benchmarks can therefore show up to ~3 % median delta for
   Ornith even though the paired, order-alternated measurement is ~0.9–1.3 %.

## Decision

**No production change.**

- Rule: if per-token gateway+SSE cost is < ~2 % of engine time for the fastest
  measured model, do not change production code.
- Fastest measured model: Ornith 9B, 63.6 tok/s (15.73–16.24 ms/token).
- Server-side SSE serialization: **2.6 µs/token = 0.016 %**.
- End-to-end HTTP delta (including client and TCP): **+0.94 %** on Ornith and
  **+0.36 %** on Bonsai in the drift-cancelled paired benchmark; the separate
  burst comparison shows 1.89 % / 0.76 % means (Ornith median 3.15 %, inflated
  by the gateway burst always running second and hotter).
- The CPU profile shows 99.4 % cgo/engine and no Go gateway hotspot; the
  smallest profile-justified optimization (single-buffer `Write` instead of
  `json.Marshal` + `fmt.Fprintf`) would save <0.02 % of engine time.
- The 7–44 % loss does **not** reproduce as gateway overhead on this host.
  The likely remaining factors are the harness/client-side measurement
  asymmetry (fixed per-token transport+consumer cost, which hurts faster models
  relatively more), differing engine build/params, and the host/GPU difference.
  The per-token bridge/detokenization hypothesis is excluded by construction.

The benchmark and this report are committed; `internal/server/handlers.go` is
unchanged.
