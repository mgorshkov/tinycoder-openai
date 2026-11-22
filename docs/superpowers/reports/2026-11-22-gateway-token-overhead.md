# Gateway per-token overhead: measurement (2026-11-22)

Task F3 of the server follow-up plan. Question: does the OpenAI-compatible HTTP
gateway add a fixed per-token cost that explains the report's 7–44 % generation
loss relative to `tinycoder_bench` (§4.7: Qwen2.5-Coder-1.5B 416 → 234 tok/s,
Qwen3.6-27B 33 → 20, Bonsai 69 → 57)?

**Answer (load-bearing evidence): no. The gateway's own serialization work is
2.6 µs per token — 0.011–0.047 % of engine token time across the measured models
(0.047 % on the fastest) — and the handler with no socket or client adds no
measurable time (Qwen −0.3 % against the table's direct mean — 705.1 vs
707.5 ms, so the sign is within process noise; the others within ±2.5 %). The
CPU profile has no Go hotspot (99.4 % `runtime.cgocall`). No production code was
changed.**

**Supporting context:** the full end-to-end HTTP path — handler + loopback
socket + in-process SSE consumer — costs more, and the fast model makes it
visible: **+2.9 %** per-request time per token for Qwen2.5-Coder-1.5B in the
paired measurement (per-count means 1.7–3.8 %, per-pair σ ≈ 5–6 %) and
**+5.7–7.4 %** per-request time in sustained single-path runs (throughput
−5.4 % to −6.9 %); Ornith 9B **+0.8 %** paired; Bonsai 27B **+0.2 %** paired.
That is the right *direction* for "the faster the model, the larger the loss",
but it is transport and client-consumption cost, not gateway/SSE serialization,
and it is 3–7 % here, not 44 %. A smaller, transport-shaped effect exists; the
report's 7–44 % does not reproduce as gateway overhead.

## Host, models, harness

| | |
|---|---|
| CPU | Intel Xeon E5-2686 v4, 72 logical CPUs |
| GPU | NVIDIA RTX 2080 Ti, 11 GB, driver 595.91.07 |
| Go | go1.26.8 linux/amd64, CGO on, native bridge `build/native/libtinycoder_bridge.so` |
| Repo | `fix/server-defects-2026-11-22` (F1/F2 fixes included) |
| Engine | tinycoder-inference through `internal/engine/native`; 32 layers offloaded, 72 engine threads |

Models (loaded once per benchmark process; never downloaded):

| id | file | size | engine-reported load | prompt tokens |
|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | `/data/models/qwen/qwen2.5/qwen2.5-coder-1.5b-instruct-q2_k.gguf` | 753 MB | weights 6.2 s + embedding dequant 3.4 s | 48 |
| Ornith 9B | `/data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf` | 5.91 GB | weights 0.4 s + embedding dequant 14.3 s | 57 |
| Bonsai 27B | `/data/models/bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5.95 GB | weights 59.8 s + embedding dequant 13.3 s | 57 |

`/data/models/gemma/gemma4-coding-Q2_K.gguf` (planned faster second model)
**fails to load** in this engine build — a model-compatibility error, not memory:

```
[TinyCoder] Missing attention weights for layer 5
[TinyCoder] Failed to load model weights
```

The Qwen2.5-Coder-1.5B Q2_K variant is the model used here; the IQ3_XXS
variant was not needed. The engine runs it at ~182 tok/s direct on this host,
not the report's 416 tok/s (different GPU/engine build/params), so the
*fractional* overhead at 416 tok/s cannot be measured directly; see the
generalization below.

Harness: `internal/server/overhead_bench_test.go` (package `server`). It skips
unless `TINYCODER_BENCH_MODEL` names a readable GGUF and is always skipped in a
`tinycoder_stub` build. The model is loaded once via
`wiring.NewFactory()("bench")` + `Load({ModelPath, MaxSeqLen: 4096})`; the
`Manager` receives a wrapper whose `Load` is a no-op and whose `Loaded` is true,
so `Manager.Acquire` takes its resident fast path and never reloads.

## Commands (exact)

```sh
# separate sustained-burst benchmarks: direct, end-to-end HTTP, handler-only
for MODEL in \
  /data/models/qwen/qwen2.5/qwen2.5-coder-1.5b-instruct-q2_k.gguf \
  /data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf \
  /data/models/bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf; do
  TINYCODER_BENCH_MODEL=$MODEL go test ./internal/server -run '^$' \
    -bench 'Benchmark(EngineDirect|GatewayStream)$' -benchtime=10x -count=5 -timeout 60m
  TINYCODER_BENCH_MODEL=$MODEL go test ./internal/server -run '^$' \
    -bench 'BenchmarkGatewayHandler$' -benchtime=10x -count=3 -timeout 30m
  TINYCODER_BENCH_MODEL=$MODEL go test ./internal/server -run '^$' \
    -bench 'BenchmarkGatewayDelta$' -benchtime=10x -count=3 -timeout 45m
done
# (Bonsai used -count=3 for the direct/gateway burst run, not -count=5.)

# pure SSE serialization, time-based so ns/op is not fixed-cost inflated by a
# 10-iteration run
TINYCODER_BENCH_MODEL=/data/models/bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  go test ./internal/server -run '^$' -bench 'BenchmarkSSEWrite$' -benchtime=2s -count=3
```

`BenchmarkGatewayDelta` runs **one direct generation and one gateway request
inside each benchmark iteration**, with the order alternating by iteration
parity. Each measured call is preceded by an untimed 1-token `resetSampler`
generation, so the two members are adjacent apart from that reset (not literally
back-to-back); both members still belong to one iteration, and the alternating
order cancels GPU clock drift. It reports the per-pair ratio distribution.
`BenchmarkGatewayHandler` runs the same request through
`http.Handler.ServeHTTP` with an `httptest.ResponseRecorder` — no TCP and no
HTTP client, `Flush` a no-op — isolating the handler's own work.

## Controls held on both paths

- Same model instance, loaded once before timing; warmed with one direct
  `Generate` and one full gateway streaming request before `b.ResetTimer()`.
- Same request: chat `Messages` (chat-template render inside both numbers;
  prompt sizes above), `MaxTokens=128`, `Temperature 0.7, TopP 0.9, TopK 40,
  RepeatPenalty 1.1, RepeatLastN 64` (the gateway defaults). The HTTP body sets
  only `model`, `messages`, `max_tokens`, `stream`, `seed`.
- Both callbacks go through the same cgo shim (`tcGoTokenCallback`), so the
  bridge's per-token transition and detokenization are inside BOTH numbers. The
  delta isolates the gateway layer, not the bridge. Sharing the shim removes the
  bridge from the gateway DELTA only: it does not measure the absolute
  server-vs-`tinycoder_bench` gap.
- Sampling at temperature 0.7 is stochastic and the engine keeps a persistent
  RNG (seed 0 seeds once from `random_device`, then continues), so the two
  paths cannot otherwise be guaranteed to emit the same number of tokens. Each
  iteration pins the same seed (cycling 1..8) in both requests; a changing seed
  forces the sampler to reseed, making the token sequence a function of
  (prompt, seed). The per-seed token count is measured once in setup and
  asserted on both paths; every measured iteration produced exactly 128
  completion tokens. The seed is not a sampling parameter under study.
- HTTP client consumes every byte up to `[DONE]`, counting non-empty
  `delta.content` values; it does no other per-token work. Direct callback only
  counts.
- `NoTokenTimeoutSec: 0` (config default), `MaxConcurrent: 1`,
  `MaxLoadedModels: 1`, discard `slog` logger (keeps access-log formatting, not
  I/O).
- The direct path calls `engine.Generate` directly, bypassing the Manager; the
  gateway path goes through `Manager.Generate`, which adds a semaphore, lease
  and a `HardTimeout` timer from config (660 s default). Both are per-request
  constants, not per-token costs, and are part of what the end-to-end number
  measures.
- The end-to-end number includes the client's read/parse and loopback TCP
  transport, because that is what a consumer of the HTTP path pays.
  `BenchmarkSSEWrite` and `BenchmarkGatewayHandler` isolate the server parts.

## Raw results

Go 1.26 prints each benchmark as a two-line progress line
(`BenchmarkX-72 ...`) followed by the metrics line because the engine writes to
stdout during the run. The lines below join the name and metrics; whitespace is
collapsed, values are verbatim.

Qwen2.5-Coder-1.5B, `-benchtime=10x -count=5` (5 direct runs, then 5 gateway
runs — `-count` repeats one benchmark fully before the next, so these are
sustained 10-iteration bursts, not paired):

```
BenchmarkEngineDirect-72   10  709465852 ns/op  180.4 tok/s  128.0 tokens/op  1579 B/op  141 allocs/op
BenchmarkEngineDirect-72   10  713857057 ns/op  179.3 tok/s  128.0 tokens/op  1580 B/op  141 allocs/op
BenchmarkEngineDirect-72   10  715383424 ns/op  178.9 tok/s  128.0 tokens/op  1580 B/op  141 allocs/op
BenchmarkEngineDirect-72   10  708647989 ns/op  180.6 tok/s  128.0 tokens/op  1628 B/op  141 allocs/op
BenchmarkEngineDirect-72   10  689873458 ns/op  185.5 tok/s  128.0 tokens/op  1580 B/op  141 allocs/op
BenchmarkGatewayStream-72  10  755678221 ns/op  169.4 tok/s  128.0 tokens/op  249672 B/op  3124 allocs/op
BenchmarkGatewayStream-72  10  769024403 ns/op  166.4 tok/s  128.0 tokens/op  243568 B/op  3117 allocs/op
BenchmarkGatewayStream-72  10  752920584 ns/op  170.0 tok/s  128.0 tokens/op  248899 B/op  3120 allocs/op
BenchmarkGatewayStream-72  10  766058509 ns/op  167.1 tok/s  128.0 tokens/op  253656 B/op  3118 allocs/op
BenchmarkGatewayStream-72  10  756868233 ns/op  169.1 tok/s  128.0 tokens/op  243520 B/op  3107 allocs/op
```

Qwen handler-only, `-benchtime=10x -count=3`:

```
BenchmarkGatewayHandler-72  10  701864722 ns/op  182.4 tok/s  128.0 tokens/op  134837 B/op  759 allocs/op
BenchmarkGatewayHandler-72  10  703239506 ns/op  182.0 tok/s  128.0 tokens/op  134828 B/op  758 allocs/op
BenchmarkGatewayHandler-72  10  710196182 ns/op  180.2 tok/s  128.0 tokens/op  134901 B/op  759 allocs/op
```

Qwen paired (`BenchmarkGatewayDelta`, `-benchtime=10x -count=3`, 10 pairs per
run; `delta-ns/op` is the mean per-pair delta, i.e. per request):

```
      10  1516925663 ns/op  26474732 delta-ns/op  +12.40 delta-pct-max   +3.756 delta-pct-mean  -7.384 delta-pct-min  5.696 delta-pct-std  723.8 direct-ms/op  750.3 gateway-ms/op  10 pairs
      10  1538932922 ns/op  11502570 delta-ns/op  +11.38 delta-pct-max   +1.680 delta-pct-mean  -7.292 delta-pct-min  6.489 delta-pct-std  740.1 direct-ms/op  751.6 gateway-ms/op  10 pairs
      10  1532554187 ns/op  26635046 delta-ns/op  +11.40 delta-pct-max   +3.747 delta-pct-mean  -1.302 delta-pct-min  5.042 delta-pct-std  731.7 direct-ms/op  758.3 gateway-ms/op  10 pairs
```

Ornith 9B, `-benchtime=10x -count=5`:

```
BenchmarkEngineDirect-72   10  2100898059 ns/op  60.93 tok/s  128.0 tokens/op  1896 B/op  141 allocs/op
BenchmarkEngineDirect-72   10  2008533706 ns/op  63.73 tok/s  128.0 tokens/op  1846 B/op  141 allocs/op
BenchmarkEngineDirect-72   10  2013199351 ns/op  63.58 tok/s  128.0 tokens/op  1980 B/op  141 allocs/op
BenchmarkEngineDirect-72   10  2081866343 ns/op  61.48 tok/s  128.0 tokens/op  1742 B/op  141 allocs/op
BenchmarkEngineDirect-72   10  2013971708 ns/op  63.56 tok/s  128.0 tokens/op  1739 B/op  141 allocs/op
BenchmarkGatewayStream-72  10  2068542892 ns/op  61.88 tok/s  128.0 tokens/op  241763 B/op  3078 allocs/op
BenchmarkGatewayStream-72  10  2085181928 ns/op  61.39 tok/s  128.0 tokens/op  237012 B/op  3075 allocs/op
BenchmarkGatewayStream-72  10  2076508741 ns/op  61.64 tok/s  128.0 tokens/op  240443 B/op  3080 allocs/op
BenchmarkGatewayStream-72  10  2089498563 ns/op  61.26 tok/s  128.0 tokens/op  247549 B/op  3080 allocs/op
BenchmarkGatewayStream-72  10  2087706197 ns/op  61.31 tok/s  128.0 tokens/op  242990 B/op  3074 allocs/op
```

Bonsai 27B, `-benchtime=10x -count=3`:

```
BenchmarkEngineDirect-72   10  3010042590 ns/op  42.52 tok/s  128.0 tokens/op  1790 B/op  141 allocs/op
BenchmarkEngineDirect-72   10  3004979227 ns/op  42.60 tok/s  128.0 tokens/op  1838 B/op  141 allocs/op
BenchmarkEngineDirect-72   10  3000942392 ns/op  42.65 tok/s  128.0 tokens/op  1691 B/op  141 allocs/op
BenchmarkGatewayStream-72  10  3035545130 ns/op  42.17 tok/s  128.0 tokens/op  240523 B/op  3080 allocs/op
BenchmarkGatewayStream-72  10  3018796188 ns/op  42.40 tok/s  128.0 tokens/op  244338 B/op  3079 allocs/op
BenchmarkGatewayStream-72  10  3030366650 ns/op  42.24 tok/s  128.0 tokens/op  251466 B/op  3084 allocs/op
```

Pure SSE serialization (`sseWriter.write` of a fixed `chatChunk` through a
discard `ResponseWriter` with `Flush`, no engine), `-benchtime=2s -count=3`:

```
BenchmarkSSEWrite-72  1000000  2586 ns/op  282 B/op  3 allocs/op
BenchmarkSSEWrite-72   884095  2651 ns/op  282 B/op  3 allocs/op
BenchmarkSSEWrite-72   858702  2620 ns/op  282 B/op  3 allocs/op
```

(At `-benchtime=10x` SSEWrite reports ~10 µs/op because ten iterations cannot
amortise the benchmark harness's fixed per-run cost; the time-based run is the
real per-frame cost.)

Auditability: raw console lines are pasted above for Qwen
direct/gateway/handler-only/paired and for the Ornith and Bonsai sustained
direct/gateway bursts. The **Ornith and Bonsai handler-only rows, the Ornith and
Bonsai paired rows, and the fresh single-path Qwen runs** are derived from the
aggregate runs; their raw console lines were not retained. The derived tables
below are consequently reproducible only at the aggregate level, not from
retained console output.

## Derived numbers

Separate runs are sustained 10-iteration bursts; they are *not* paired across
paths (`-count` completes all runs of one benchmark before starting the next),
so only means/ranges are meaningful here. Sorted medians are shown for
completeness.

| model | path | mean ms/op | mean tok/s | sorted median ms | sorted median tok/s | range tok/s |
|---|---|---|---|---|---|---|
| Qwen 1.5B | direct | 707.5 | 180.9 | 709.5 | 180.4 | 178.9–185.5 |
| Qwen 1.5B | gateway | 760.1 | 168.4 | 756.9 | 169.1 | 166.4–170.0 |
| Qwen 1.5B | handler only | 705.1 | 181.5 | 703.2 | 182.0 | 180.2–182.4 |
| Ornith 9B | direct | 2043.7 | 62.6 | 2014.0 | 63.6 | 60.9–63.7 |
| Ornith 9B | gateway | 2081.5 | 61.5 | 2085.2 | 61.4 | 61.3–61.9 |
| Ornith 9B | handler only | 2014.2 | 63.6 | 2009.9 | 63.7 | 62.7–64.3 |
| Bonsai 27B | direct | 3005.3 | 42.6 | 3005.0 | 42.6 | 42.5–42.7 |
| Bonsai 27B | gateway | 3028.2 | 42.3 | 3030.4 | 42.2 | 42.2–42.4 |
| Bonsai 27B | handler only | 3081.0 | 41.5 | 3111.2 | 41.1 | 41.1–42.4 |

Direct-path note: the handler-only path is indistinguishable from direct on all
three models. Against each model's separate-run direct mean, Qwen is −0.3 %
(handler 705.1 ms vs direct 707.5 ms; using the fresh single-path 703.2 ms
baseline instead flips the sign to +0.3 %, which shows the comparison is below
the process noise floor), Ornith is −1.4 %, and Bonsai is +2.5 %
(process-to-process GPU variance is ±2–3 %, so all three are noise). That is the
server handler's own cost: request decode, preflight, manager, and
`json.Marshal`+`Fprintf` per token, with no socket and a no-op `Flush`. It does
not explain the end-to-end delta.

The paired direct runs differ from these separate direct runs by up to ~3 %
(Qwen paired direct ≈ 732 ms vs 707 ms separate; Bonsai 3088–3102 ms vs
3005 ms). The paired loop keeps the GPU continuously busy and each measured
call follows a 1-token seed reset, so it runs at the sustained-clock level;
neither number is wrong, but the paired ratio uses its own direct denominator.

## Paired per-iteration measurement (end-to-end delta)

`BenchmarkGatewayDelta`, `-benchtime=10x -count=3` = 3 × 10 pairs, each pair
one direct and one gateway call. Per-pair ratio = (gateway − direct)/direct for
the same seed and iteration; the ratio's sign is always this definition, while
the ORDER of the two calls alternates by iteration parity. Each call is preceded
by the untimed 1-token `resetSampler` described above. `delta-pct-mean/std/min/max`
are the statistics over the 10 pairs of each run:

| model | run | mean | std | min | max | aggregate Δ ms/request | aggregate % |
|---|---|---|---|---|---|---|---|
| Qwen 1.5B | 1 | +3.76 % | 5.70 % | −7.38 % | +12.40 % | +26.47 | |
| Qwen 1.5B | 2 | +1.68 % | 6.49 % | −7.29 % | +11.38 % | +11.50 | |
| Qwen 1.5B | 3 | +3.75 % | 5.04 % | −1.30 % | +11.40 % | +26.64 | |
| Qwen 1.5B | all | | | | | +21.54 | **+2.94 %** |
| Ornith 9B | 1 | +1.62 % | 1.52 % | −1.14 % | +3.78 % | +32.05 | |
| Ornith 9B | 2 | +0.80 % | 1.18 % | −0.77 % | +3.32 % | +16.10 | |
| Ornith 9B | 3 | −0.06 % | 2.15 % | −2.84 % | +3.50 % | −1.51 | |
| Ornith 9B | all | | | | | +15.55 | **+0.77 %** |
| Bonsai 27B | 1 | +0.28 % | 0.77 % | −0.92 % | +1.58 % | +8.62 | |
| Bonsai 27B | 2 | −0.27 % | 1.02 % | −2.08 % | +0.79 % | −8.50 | |
| Bonsai 27B | 3 | +0.66 % | 0.89 % | −0.13 % | +2.81 % | +20.33 | |
| Bonsai 27B | all | | | | | +6.82 | **+0.22 %** |

(The aggregate % is Δ over the sum of direct means, i.e. the ratio of the
per-pair delta totals; it is not the average of the three per-run means. The
Qwen rows' raw console lines are pasted above; the Ornith and Bonsai rows are
derived from the aggregate runs — raw lines not retained.)

At the fast model the per-pair spread is large (σ ≈ 5–6 %, single pairs from
−7 % to +12 %), so the mean is not tightly determined: the three run means are
1.7–3.8 % and the aggregate is 2.9 %. The data do **not** support a claim that
the end-to-end path is comfortably below 2 % on Qwen; they do support that the
gateway's own handler/serialization is far below it.

Sustained-burst comparisons (direct burst mean vs gateway burst mean, all
counts of one path before the other, order/thermal effects included):

| model | direct ms | gateway ms | Δ per request | Δ % per-request time | throughput change |
|---|---|---|---|---|---|
| Qwen 1.5B | 707.5 | 760.1 | +52.7 ms | +7.4 % | −6.9 % (180.9 → 168.4 tok/s) |
| Qwen 1.5B (fresh single-path processes) | 703.2 | 743.1 | +39.9 ms | +5.7 % | −5.3 % (182.0 → 172.3 tok/s) |
| Ornith 9B | 2043.7 | 2081.5 | +37.8 ms | +1.9 % | −1.8 % (62.6 → 61.5 tok/s) |
| Bonsai 27B | 3005.3 | 3028.2 | +22.9 ms | +0.8 % | −0.7 % (42.6 → 42.3 tok/s) |

The Δ % column is per-request time (positive = slower); the last column is the
corresponding throughput loss. The two are not interchangeable.

## Decomposition

| model | engine direct ms/token (separate burst) | SSE `write` + `Flush` per token | handler-only vs direct mean (separate burst) | end-to-end paired delta (paired direct baseline) | sustained burst delta (separate direct baseline) |
|---|---|---|---|---|---|
| Qwen 1.5B | 5.53 (180.9 tok/s) | 2.62 µs = **0.047 %** | −0.3 % (705.1 vs 707.5 ms) | +0.168 ms = +2.9 % | +0.31 ms = +5.7 % (fresh single-path) |
| Ornith 9B | 15.97 (62.6 tok/s) | 2.62 µs = **0.016 %** | −1.4 % (noise) | +0.121 ms = +0.8 % | +0.30 ms = +1.9 % |
| Bonsai 27B | 23.48 (42.6 tok/s) | 2.62 µs = **0.011 %** | +2.5 % (noise) | +0.053 ms = +0.2 % | +0.18 ms = +0.8 % |

The baselines are not the same across columns. The engine-direct and
handler-only columns use the table's separate-burst direct mean (Qwen
5.53 ms/token, i.e. 707.5 ms per 128-token request). The sustained column uses
each row's stated separate-run direct baseline — for Qwen the fresh single-path
703.2 ms (5.49 ms/token) for the +5.7 % cell, or 707.5 ms (5.53 ms/token) for
the +7.4 % combined run. The paired column instead uses the paired direct mean
(Qwen ≈5.72 ms/token, 732 ms/request), so `0.168 ms / 5.72 ms = 2.9 %`, not
`0.168 / 5.53 = 3.0 %`. The SSE share uses each model's *own* engine token time,
so it is larger for the fast model in percentage terms while still under
0.05 %. Everything else in the end-to-end delta is outside the handler: socket
transport, the in-process client's read/parse/wake-ups, and scheduling
contention with the engine's 72 threads. Two caveats matter: (a) the client in
this harness is in the same process, so part of that cost would be remote in a
real deployment; (b) `sseWriter.write` and `Flush` run synchronously inside the
engine's token callback on the generation thread, so a *slow remote consumer*
can still back-pressure and throttle generation — this loopback harness cannot
measure that case, and the 2.6 µs figure is the serialization cost only, not a
guarantee about a slow consumer.

## CPU profile

Exact commands (compiled binary so symbols resolve; profile includes setup,
which is why `tc_engine_generate` is 63 s):

```sh
go test -c -o /tmp/f3-server.test ./internal/server
TINYCODER_BENCH_MODEL=/data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf \
  /tmp/f3-server.test -test.run '^$' -test.bench 'BenchmarkGatewayStream$' \
  -test.benchtime=20x -test.cpuprofile=/tmp/f3-gw.prof -test.timeout 30m
go tool pprof -top -nodecount=20 /tmp/f3-server.test /tmp/f3-gw.prof
```

Budget: 82.1 s duration, 82.21 s samples. It covers the model load (18.7 s
`tc_engine_load`) and all `tc_engine_generate` calls: 8 untimed setup
generations plus 20 timed gateway requests, i.e. ~63 s total. The timed region
alone is ~42 s; no Go gateway function reaches the 0.4 s sample cutoff.

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

Supplemental SSEWrite profile (1 M frames): `encoding/json.structEncoder.encode`
44.7 % cum, `fmt.Fprintf` 7.9 %; total 2.6 µs/frame, so pre-serializing a frame
could save at most ~1 µs/token — 0.0064 % of engine time at 64 tok/s
(1 µs / 15.6 ms).

## Comparison with the report (§4.7)

| report data point (`tinycoder_bench` vs server) | this measurement (engine direct vs HTTP gateway) |
|---|---|
| Qwen2.5-Coder-1.5B 416 → 234 tok/s (−44 %) | **Qwen2.5-Coder-1.5B Q2_K: 180.9 → 168.4 tok/s sustained (−6.9 % throughput; +7.4 % per-request time), +2.9 % paired time** on this host |
| Qwen3.6-27B 33 → 20 tok/s (−39 %) | not available locally and not reproduced; conclusion is by generalization from the per-token serialization cost (0.011–0.016 %) and the measured fast/slow models |
| Bonsai 69 → 57 tok/s (−17 %) | 42.6 → 42.3 tok/s sustained (−0.7 % throughput; +0.8 % per-request time), +0.2 % paired time |
| faster model ⇒ larger loss | reproduced in *direction* (per-request time: Qwen +2.9 % paired / +7.4 % sustained, Ornith +0.8 %, Bonsai +0.2 % paired; throughput losses −6.9 % / −1.8 % / −0.7 % respectively), but it is transport/client cost, and the magnitude is 3–7 %, not 44 % |

Factors that differ; they were controlled or excluded here:

1. **Different harness on the baseline side.** `tinycoder_bench` calls the
   engine directly; the server number is an HTTP/SSE client measurement. Any
   fixed per-token consumer/transport cost is divided by a shrinking per-token
   engine time as models get faster. This is the one factor our data confirms
   *directionally*. In the paired runs the extra per-token time is
   0.053–0.168 ms (0.2 % at 42 tok/s, 0.8 % at 63, 2.9 % at 181, measured
   against each pair's own direct baseline); in the sustained-burst runs it is
   0.18–0.31 ms (0.8 % at 42, 1.9 % at 63, 5.7 % at 181, measured against the
   separate direct burst). At the report's 416 tok/s (2.4 ms/token) the
   sustained per-token cost would be ~7.5–13 % — meaningful, but still far from
   44 %.
2. **Different request semantics.** Here both sides render chat `Messages`
   through the model's chat template (prompt sizes in the model table), so the
   template is not a server-side penalty. `tinycoder_bench` may use a raw
   prompt, different sampler values, a different `max_tokens`, or a different
   measurement window (steady-state generation vs a whole short request
   including prefill/first token).
3. **Different engine build/configuration.** This branch contains the F1/F2
   fixes; the report predates them. Thread count/offload split, engine version
   and `n_threads` differ (this host runs 72 engine threads on 72 vCPUs).
4. **Different host.** RTX 2080 Ti 11 GB vs the report's 3090. Absolute
   throughput is lower here (Qwen 1.5B 181 tok/s direct vs the report's 416;
   Bonsai 42.6 vs 69), so absolute numbers are not comparable; the ratio is
   what this task tests.
5. **Per-token bridge/cgo is removed from the gateway delta by construction.**
   Both paths here use the same `tcGoTokenCallback`/shim, so detokenization and
   the cgo transition are inside both numbers and cancel in the
   gateway-vs-direct delta. That does not measure the absolute
   server-vs-`tinycoder_bench` gap, whose baseline path may differ; bridge cost
   is therefore not ruled out as a component of that gap.
6. **Run-to-run noise.** Engine time varies ±2–3 % per 10-iteration burst on
   this host (GPU clock), which is larger than the small-model deltas; the
   paired benchmark exists to cancel it, and its per-pair spread is reported
   above.

## Decision

**No production change.**

- Load-bearing evidence: the server's own per-token gateway work is tiny. SSE
  serialization is **2.6 µs/token (0.011–0.047 %)**; the handler with no socket
  or client is **indistinguishable from direct** (Qwen −0.3 % against the
  separate-burst direct mean — 705.1 vs 707.5 ms, i.e. the sign is inside
  process noise; Ornith −1.4 %, Bonsai +2.5 %; all within ±2–3 % process noise);
  the CPU profile is 99.4 % cgo/engine with no Go hotspot; the
  profile-justified serialization optimization would save <0.02 % of engine time
  (~0.0064 % at 64 tok/s).
- Supporting context: the end-to-end HTTP path (handler + loopback transport +
  in-process client) costs +0.2 % (Bonsai) to +2.9 % (Qwen) per-request time
  paired, and up to +5.7–7.4 % per-request time in sustained runs at 181 tok/s
  (throughput −5.4 % to −6.9 %). This is above the ~2 % threshold for the fast
  model, but it is **outside the handler** — it is socket transport, client
  consumption and scheduling, not the SSE serialization the task could
  optimize. The decision rule's threshold applies to gateway+SSE work, which is
  0.016–0.047 %.
- The report's 7–44 % loss does not reproduce as gateway/SSE overhead. A
  smaller, model-speed-dependent transport/consumer effect is real (~3–7 % at
  181 tok/s here). The remaining difference must be measured on the report's
  harness/host: different benchmark params/prompt-output mix, an engine build
  without the F1/F2 fixes, thread/offload configuration, or client-side/network
  overhead not represented by a loopback consumer. Sharing the cgo shim
  excludes detokenization and the bridge from the gateway delta, but it does not
  measure the absolute server-vs-`tinycoder_bench` gap (the baseline path may
  differ); the residual is also not `sseWriter.write` (2.6 µs/token).
- If streaming cost ever needs to come down, the lever is the delivery path
  (flush batching / avoiding one syscall per token, or measuring a remote
  consumer), not pre-serializing the SSE frame; that is out of scope for this
  task and not justified by the profile.

`internal/server/handlers.go` is unchanged.

## Revision note (fix round 1)

- Added the Qwen2.5-Coder-1.5B Q2_K measurement (the report's decisive fast
  case) and removed the "not available locally" claim.
- Paired benchmark now reports per-pair mean/std/min/max and was run with
  `-count=3`; the "comfortably below 2 %" wording is dropped for the end-to-end
  path (it is ~2.9 % on Qwen) and the serialization share is stated as the
  load-bearing evidence.
- Fixed the run-order description: `-count` runs all counts of one benchmark
  before the next, so direct/gateway burst runs are not paired; per-count
  "deltas" were removed in favour of means/ranges plus the paired benchmark.
- Added a committed `BenchmarkGatewayHandler` (ResponseRecorder, no TCP/client)
  so the transport-vs-handler attribution is reproducible; the earlier
  uncommitted probe is no longer cited as load-bearing.
- Corrected Ornith sorted medians; reworded the `NoTokenTimeoutSec` comment;
  recorded the exact `-cpuprofile` command and its 82 s budget (including the 8
  untimed setup generations); tightened the remote-consumer caveat; noted that
  the direct path bypasses the Manager (different `HardTimeout`) and why paired
  direct and separate direct differ by ~3 %.
- Raw results now carry benchmark-name prefixes (joined from Go's two-line
  progress output, whitespace collapsed).

## Revision note (fix round 2, documentation accuracy)

- Corrected the paired-ratio description: the ORDER alternates by parity; the
  ratio's sign is always `(gateway − direct)/direct`.
- Labeled time vs throughput everywhere: Qwen's throughput loss is −6.9 %
  (180.9 → 168.4 tok/s), Bonsai's −0.7 % (42.6 → 42.3); the +7.4 % / +0.8 %
  figures are per-request time increases.
- Fixed the Qwen handler-only sign and stated its baseline (−0.3 % against the
  705.1 vs 707.5 ms direct mean; +0.3 % against the fresh 703.2 ms baseline, so
  the comparison is below the noise floor).
- Fixed the mixed-denominator arithmetic in the transport-factor discussion:
  paired deltas are 0.053–0.168 ms, sustained deltas 0.18–0.31 ms, each stated
  against its own baseline.
- Corrected the serialization-optimization estimate to ~0.0064 % at 64 tok/s
  (1 µs / 15.6 ms).
- Named each decomposition column's baseline (separate-burst direct vs paired
  direct ≈5.72 ms/token) so the paired 2.9 % is not read against the separate
  5.53 ms/token.
- Noted the untimed 1-token `resetSampler` between paired members (both remain
  in one iteration) and reworded the bridge/cgo claim: sharing the shim removes
  it from the gateway delta only, not from the unmeasured absolute
  server-vs-`tinycoder_bench` gap.
- Added Qwen paired raw console lines and an explicit auditability note for the
  Ornith/Bonsai handler-only and paired rows and the fresh single-path Qwen runs
  (derived from the aggregate runs; raw lines not retained).
