package server

// Gateway per-token overhead benchmarks.
//
// These benchmarks exist to answer one question with numbers instead of
// hypotheses: how much wall-clock time does the HTTP/SSE gateway add per
// generated token, compared with calling the native engine directly?
//
// They are opt-in and model-backed: set TINYCODER_BENCH_MODEL to a real local
// GGUF and run `go test ./internal/server -bench 'Benchmark(EngineDirect|
// GatewayStream|GatewayDelta|SSEWrite)$' -run '^$'`. Without the variable (and
// always in a `tinycoder_stub` build, which has no real engine) every benchmark
// skips, so the default `go test ./...` stays fast and model-free.
//
// Fairness controls:
//   - The model is loaded ONCE per process and shared by both the direct and
//     gateway benchmarks. The Manager gets a factory whose engine never loads
//     and always reports Loaded, so Manager.Acquire takes its resident fast
//     path instead of reloading the 5 GB model.
//   - Both timed paths build the same engine.GenerateRequest: chat Messages
//     (so the chat template render is inside both numbers), MaxTokens=128, and
//     the gateway's default sampling parameters (temperature 0.7, top_p 0.9,
//     top_k 40, repeat penalty 1.1, repeat last N 64).
//   - The model is warmed once before any timing: one direct Generate call and
//     one full gateway streaming request.
//   - Sampling is stochastic at temperature 0.7 and the engine's sampler keeps
//     a persistent RNG across calls (seed 0 means "seed from random_device
//     once"), so two runs of the same request do not necessarily emit the same
//     number of tokens. To compare like with like, each iteration pins a seed
//     (cycling 1..8) in BOTH requests: a new seed forces the sampler to reseed,
//     so the token sequence depends only on (prompt, seed). The expected token
//     count for every seed is measured once in setup and asserted on both
//     paths. The seed is not a sampling parameter under study; the gateway's
//     default temperature/top_p/top_k/repeat values are unchanged.
//   - The HTTP client consumes every byte of the SSE body and counts non-empty
//     delta.content values; it does no other per-token work. The direct path's
//     token callback only counts.
//   - Both paths assert that they generated the same number of completion
//     tokens; a mismatch fails the benchmark rather than silently skewing ns/op.
//   - The engine's token callback goes through the same cgo shim in both paths,
//     so the bridge's per-token transition is inside BOTH numbers. The
//     difference isolates the gateway/SSE layer, not the bridge.
//   - BenchmarkGatewayDelta is a paired cross-check: it measures direct and
//     gateway back-to-back for the same seed (alternating order), because GPU
//     clock drift between two separate timed bursts is larger than the delta,
//     and reports the per-pair ratio distribution.
//   - BenchmarkGatewayHandler runs the same request through the handler with a
//     ResponseRecorder (no TCP, no HTTP client), isolating the gateway's own
//     serialization work from socket transport and client consumption.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mgorshkov/tinycoder-openai/internal/config"
	"github.com/mgorshkov/tinycoder-openai/internal/engine"
	"github.com/mgorshkov/tinycoder-openai/internal/wiring"
)

const (
	// benchModelEnv names the local GGUF the overhead benchmarks run against.
	benchModelEnv = "TINYCODER_BENCH_MODEL"
	// benchModelID is the registry id the benchmarks expose to the gateway.
	benchModelID = "bench"
	// benchPrompt is a fixed instruction whose answer cannot fit in 128 tokens,
	// so most runs are truncated by max_tokens instead of stopping early on
	// EOS. The exact completion length does not matter for the comparison: it
	// is measured per seed and asserted to be identical on both paths, no
	// matter where the model stops.
	benchPrompt = "Write a complete Go program that implements a thread-safe LRU cache with unit tests. The program must be at least 300 lines and include all imports, comments, and every function."
	// benchMaxTokens is the generation cap for every benchmark path.
	benchMaxTokens = 128
	// benchContextLen is the context window used for the load and the registry.
	benchContextLen = 4096
	// benchTokenText is a representative decoded token delta for the pure SSE
	// serialization benchmark.
	benchTokenText = " hash"
	// benchSeedCount is the number of per-iteration seeds cycled through. A
	// distinct seed per iteration forces the sampler to reseed on every call,
	// which makes each iteration reproducible in both paths.
	benchSeedCount = 8
	// benchResetSeed is used by one untimed generation before each benchmark's
	// timed loop to move the persistent RNG off the first cycle seed, so the
	// first timed call reseeds too. It is deliberately outside the 1..8 cycle.
	benchResetSeed = 0xBEEF
)

// benchShared holds the process-wide, load-once benchmark fixtures.
var benchShared struct {
	once sync.Once
	env  *benchEnv
	err  error
}

// benchEnv is the shared model, manager and HTTP server for one benchmark
// process.
type benchEnv struct {
	engine  engine.Engine
	manager *Manager
	server  *httptest.Server
	client  *http.Client
	// bodies[seed-1] is the pre-marshaled streaming chat request for that seed,
	// so per-iteration client-side JSON encoding is not part of the measured
	// gateway path.
	bodies [][]byte
	// expected[seed-1] is the completion token count produced by a direct
	// generation with that seed; both timed paths assert it.
	expected []int
}

// Benchmarks require a real local model and the native engine; otherwise skip.
func benchGuard(tb testing.TB) string {
	tb.Helper()
	if !wiring.Native {
		tb.Skip("tinycoder_stub build: the gateway overhead benchmarks need the native engine")
	}
	raw := strings.TrimSpace(os.Getenv(benchModelEnv))
	if raw == "" {
		tb.Skipf("%s is not set; point it at a local GGUF to run the gateway overhead benchmarks", benchModelEnv)
	}
	abs, err := filepath.Abs(raw)
	if err != nil {
		tb.Fatalf("resolve %s=%q: %v", benchModelEnv, raw, err)
	}
	fi, err := os.Stat(abs)
	if err != nil {
		tb.Skipf("%s=%q is not readable: %v", benchModelEnv, raw, err)
	}
	if !fi.Mode().IsRegular() {
		tb.Skipf("%s=%q is not a regular GGUF file", benchModelEnv, raw)
	}
	return abs
}

// benchSetup returns the shared benchmark fixtures, loading the model once per
// process on first use.
func benchSetup(tb testing.TB) *benchEnv {
	tb.Helper()
	path := benchGuard(tb)
	benchShared.once.Do(func() {
		benchShared.env, benchShared.err = newBenchEnv(path)
	})
	if benchShared.err != nil {
		tb.Fatalf("gateway overhead benchmark setup: %v", benchShared.err)
	}
	return benchShared.env
}

// preloadedEngine adapts the already-loaded shared engine to the Manager. Load
// is a no-op after the first successful load (the model is loaded exactly once
// per process, before the Manager exists) and Loaded reports the real state so
// Manager.Acquire takes its resident fast path and never reloads the model.
type preloadedEngine struct {
	engine.Engine
}

func (p *preloadedEngine) Load(ctx context.Context, opts engine.LoadOptions) error {
	if p.Engine.Loaded() {
		return nil
	}
	return p.Engine.Load(ctx, opts)
}

func (p *preloadedEngine) Loaded() bool { return p.Engine.Loaded() }

// newBenchEnv loads the model once and builds the gateway over it.
func newBenchEnv(modelPath string) (*benchEnv, error) {
	eng := wiring.NewFactory()(benchModelID)
	// A cold load can take minutes on this host; bound it generously so a stuck
	// model surfaces as an error instead of hanging the test binary.
	loadCtx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()
	if err := eng.Load(loadCtx, engine.LoadOptions{ModelPath: modelPath, MaxSeqLen: benchContextLen}); err != nil {
		return nil, fmt.Errorf("load %q: %w", modelPath, err)
	}

	// Measure the completion token count for every cycle seed once. These runs
	// also warm the model.
	expected := make([]int, benchSeedCount)
	for seed := 1; seed <= benchSeedCount; seed++ {
		n, err := benchGenerateDirect(eng, uint32(seed))
		if err != nil {
			return nil, fmt.Errorf("warm-up direct generate (seed %d): %w", seed, err)
		}
		if n == 0 {
			return nil, fmt.Errorf("warm-up direct generate (seed %d) produced no tokens", seed)
		}
		expected[seed-1] = n
	}

	cfg := config.Default()
	cfg.ModelsDir = filepath.Dir(modelPath)
	cfg.Models = []config.RegistryModel{{
		ID:            benchModelID,
		Name:          benchModelID,
		File:          modelPath,
		ContextLength: benchContextLen,
	}}
	cfg.MaxLoadedModels = 1
	cfg.MaxConcurrent = 1
	// Disable the inactivity watchdog: it is a timing guard, not part of the
	// per-token path. 0 is the config default (config.Default() sets it to 0).
	cfg.NoTokenTimeoutSec = 0

	// A discard logger keeps the per-request access log from dominating the
	// measured HTTP path while preserving its formatting cost.
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	factory := func(string) engine.Engine { return &preloadedEngine{Engine: eng} }
	mgr := NewManager(cfg, factory, log)
	srv := httptest.NewServer(New(cfg, mgr, log).Handler())

	bodies := make([][]byte, benchSeedCount)
	for seed := 1; seed <= benchSeedCount; seed++ {
		body, err := benchChatBody(uint32(seed))
		if err != nil {
			srv.Close()
			return nil, err
		}
		bodies[seed-1] = body
	}

	env := &benchEnv{
		engine:   eng,
		manager:  mgr,
		server:   srv,
		client:   &http.Client{Transport: &http.Transport{DisableCompression: true}},
		bodies:   bodies,
		expected: expected,
	}

	// Warm the model through the full gateway path too, so the first timed
	// gateway iteration is not the one paying for the cold chat-template/tokenizer
	// caches or the manager's load branch.
	got, err := env.streamOnce(bodies[0])
	if err != nil {
		srv.Close()
		return nil, fmt.Errorf("warm-up gateway request: %w", err)
	}
	if got != expected[0] {
		srv.Close()
		return nil, fmt.Errorf("warm-up token mismatch: gateway streamed %d content deltas, direct generate produced %d tokens", got, expected[0])
	}
	return env, nil
}

// benchChatBody builds the streaming chat-completions request for a seed.
func benchChatBody(seed uint32) ([]byte, error) {
	payload := struct {
		Model     string              `json:"model"`
		Messages  []map[string]string `json:"messages"`
		MaxTokens int                 `json:"max_tokens"`
		Stream    bool                `json:"stream"`
		Seed      uint32              `json:"seed"`
	}{
		Model:     benchModelID,
		Messages:  []map[string]string{{"role": "user", "content": benchPrompt}},
		MaxTokens: benchMaxTokens,
		Stream:    true,
		Seed:      seed,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	return body, nil
}

// benchGenerateRequest is the request both timed paths use.
func benchGenerateRequest(seed uint32) engine.GenerateRequest {
	return engine.GenerateRequest{
		Model:         benchModelID,
		Messages:      []engine.Message{{Role: "user", Content: benchPrompt}},
		MaxTokens:     benchMaxTokens,
		Temperature:   0.7,
		TopP:          0.9,
		TopK:          40,
		RepeatPenalty: 1.1,
		RepeatLastN:   64,
		Seed:          seed,
	}
}

// benchGenerateDirect runs one direct generation and returns the number of
// tokens its callback saw.
func benchGenerateDirect(eng engine.Engine, seed uint32) (int, error) {
	tokens := 0
	if _, err := eng.Generate(context.Background(), benchGenerateRequest(seed), func(engine.Token) bool {
		tokens++
		return true
	}); err != nil {
		return tokens, err
	}
	return tokens, nil
}

// resetSampler runs one untimed one-token generation with benchResetSeed. The
// engine's sampler keeps its RNG across calls and only reseeds when a call's
// seed differs from the previous call's seed, so this guarantees the first
// timed call (seed 1) actually reseeds, no matter how the previous benchmark
// invocation ended.
func (e *benchEnv) resetSampler() error {
	req := benchGenerateRequest(benchResetSeed)
	req.MaxTokens = 1
	_, err := e.engine.Generate(context.Background(), req, nil)
	return err
}

// streamOnce performs one full streaming chat request and consumes the SSE
// body to [DONE], counting non-empty delta.content values.
func (e *benchEnv) streamOnce(body []byte) (int, error) {
	req, err := http.NewRequest(http.MethodPost, e.server.URL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := e.client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return 0, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}

	br := bufio.NewReader(resp.Body)
	tokens := 0
	for {
		line, readErr := br.ReadString('\n')
		if strings.HasPrefix(line, "data: ") {
			payload := strings.TrimSuffix(strings.TrimPrefix(line, "data: "), "\n")
			if payload == "[DONE]" {
				return tokens, nil
			}
			var chunk chatChunk
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				return tokens, fmt.Errorf("decode SSE frame %q: %w", payload, err)
			}
			for _, choice := range chunk.Choices {
				if choice.Delta.Content != "" {
					tokens++
				}
			}
		}
		if readErr == io.EOF {
			return tokens, errors.New("SSE stream ended without [DONE]")
		}
		if readErr != nil {
			return tokens, readErr
		}
	}
}

// reportThroughput reports the completion-token throughput derived from the
// measured elapsed time of this benchmark invocation.
func reportThroughput(b *testing.B, tokens int) {
	if b.N > 0 {
		b.ReportMetric(float64(tokens)/float64(b.N), "tokens/op")
	}
	if elapsed := b.Elapsed(); elapsed > 0 {
		b.ReportMetric(float64(tokens)/elapsed.Seconds(), "tok/s")
	}
}

// BenchmarkEngineDirect measures the native engine's Generate call in
// isolation: same request, same token callback path (through the cgo shim),
// no HTTP, no JSON, no SSE. This is the baseline the gateway is compared to.
func BenchmarkEngineDirect(b *testing.B) {
	env := benchSetup(b)
	if err := env.resetSampler(); err != nil {
		b.Fatalf("reset sampler: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	tokens := 0
	for i := 0; i < b.N; i++ {
		seed := uint32(i%benchSeedCount) + 1
		n := 0
		if _, err := env.engine.Generate(context.Background(), benchGenerateRequest(seed), func(engine.Token) bool {
			n++
			return true
		}); err != nil {
			b.Fatalf("iteration %d (seed %d): generate: %v", i, seed, err)
		}
		if want := env.expected[seed-1]; n != want {
			b.Fatalf("iteration %d (seed %d): direct generate produced %d tokens, setup measured %d", i, seed, n, want)
		}
		tokens += n
	}
	reportThroughput(b, tokens)
}

// BenchmarkGatewayStream measures the same generation end-to-end through the
// real HTTP handler: POST /v1/chat/completions with stream=true, consuming the
// SSE body to [DONE]. The difference from BenchmarkEngineDirect is the gateway
// layer (request decode, preflight, token→SSE serialization, flush, HTTP
// transport) plus the client's SSE consumption loop.
func BenchmarkGatewayStream(b *testing.B) {
	env := benchSetup(b)
	if err := env.resetSampler(); err != nil {
		b.Fatalf("reset sampler: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	tokens := 0
	for i := 0; i < b.N; i++ {
		seed := uint32(i%benchSeedCount) + 1
		n, err := env.streamOnce(env.bodies[seed-1])
		if err != nil {
			b.Fatalf("iteration %d (seed %d): %v", i, seed, err)
		}
		if want := env.expected[seed-1]; n != want {
			b.Fatalf("iteration %d (seed %d): gateway streamed %d content deltas, direct generate produced %d tokens", i, seed, n, want)
		}
		tokens += n
	}
	reportThroughput(b, tokens)
}

// BenchmarkGatewayDelta is a paired cross-check for the two separate
// benchmarks above. Engine-direct and gateway-stream run as separate timed
// bursts, and GPU-clock drift between bursts on this host (±2–3 % per 128-token
// request) is larger than the gateway delta, so the two bursts cannot resolve
// it on their own. This benchmark measures one direct generation and one
// gateway request back-to-back for the same seed, alternating which side runs
// first, so drift cancels. The benchmark's own ns/op covers both calls and is
// not meaningful; read delta-ns/op, gateway-delta-% and the per-path ms/op
// metrics instead.
func BenchmarkGatewayDelta(b *testing.B) {
	env := benchSetup(b)
	b.ResetTimer()

	measureDirect := func(seed uint32) (time.Duration, error) {
		if err := env.resetSampler(); err != nil {
			return 0, err
		}
		start := time.Now()
		n, err := benchGenerateDirect(env.engine, seed)
		elapsed := time.Since(start)
		if err != nil {
			return elapsed, err
		}
		if want := env.expected[seed-1]; n != want {
			return elapsed, fmt.Errorf("direct generate produced %d tokens, setup measured %d", n, want)
		}
		return elapsed, nil
	}
	measureGateway := func(seed uint32) (time.Duration, error) {
		if err := env.resetSampler(); err != nil {
			return 0, err
		}
		start := time.Now()
		n, err := env.streamOnce(env.bodies[seed-1])
		elapsed := time.Since(start)
		if err != nil {
			return elapsed, err
		}
		if want := env.expected[seed-1]; n != want {
			return elapsed, fmt.Errorf("gateway streamed %d content deltas, direct generate produced %d tokens", n, want)
		}
		return elapsed, nil
	}

	var (
		directTotal, gatewayTotal time.Duration
		ratios                    = make([]float64, 0, b.N)
	)
	for i := 0; i < b.N; i++ {
		seed := uint32(i%benchSeedCount) + 1
		var d, g time.Duration
		var err error
		if i%2 == 0 {
			if d, err = measureDirect(seed); err != nil {
				b.Fatalf("iteration %d (seed %d) direct: %v", i, seed, err)
			}
			if g, err = measureGateway(seed); err != nil {
				b.Fatalf("iteration %d (seed %d) gateway: %v", i, seed, err)
			}
		} else {
			if g, err = measureGateway(seed); err != nil {
				b.Fatalf("iteration %d (seed %d) gateway: %v", i, seed, err)
			}
			if d, err = measureDirect(seed); err != nil {
				b.Fatalf("iteration %d (seed %d) direct: %v", i, seed, err)
			}
		}
		directTotal += d
		gatewayTotal += g
		if d > 0 {
			ratios = append(ratios, 100*float64(g-d)/float64(d))
		}
	}
	if b.N > 0 {
		n := float64(b.N)
		b.ReportMetric(float64(directTotal.Nanoseconds())/n/1e6, "direct-ms/op")
		b.ReportMetric(float64(gatewayTotal.Nanoseconds())/n/1e6, "gateway-ms/op")
		b.ReportMetric(float64((gatewayTotal-directTotal).Nanoseconds())/n, "delta-ns/op")
		if len(ratios) > 0 {
			mean, std, lo, hi := meanStdMinMax(ratios)
			b.ReportMetric(mean, "delta-pct-mean")
			b.ReportMetric(std, "delta-pct-std")
			b.ReportMetric(lo, "delta-pct-min")
			b.ReportMetric(hi, "delta-pct-max")
			b.ReportMetric(float64(len(ratios)), "pairs")
		}
	}
}

// meanStdMinMax returns the sample mean, sample standard deviation, minimum and
// maximum of v.
func meanStdMinMax(v []float64) (mean, std, lo, hi float64) {
	lo, hi = v[0], v[0]
	for _, x := range v {
		mean += x
		if x < lo {
			lo = x
		}
		if x > hi {
			hi = x
		}
	}
	mean /= float64(len(v))
	if len(v) > 1 {
		var ss float64
		for _, x := range v {
			d := x - mean
			ss += d * d
		}
		std = math.Sqrt(ss / float64(len(v)-1))
	}
	return mean, std, lo, hi
}

// BenchmarkGatewayHandler runs the streaming chat request through the HTTP
// handler in-process with an httptest.ResponseRecorder: no TCP socket and no
// HTTP client, and Recorder.Flush is a no-op. It isolates the gateway's own
// per-token work (request decode, preflight, manager, token→SSE serialization)
// from socket transport and client consumption, and is the committed version of
// the attribution probe used in the report.
func BenchmarkGatewayHandler(b *testing.B) {
	env := benchSetup(b)
	handler := env.server.Config.Handler
	if err := env.resetSampler(); err != nil {
		b.Fatalf("reset sampler: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	tokens := 0
	for i := 0; i < b.N; i++ {
		seed := uint32(i%benchSeedCount) + 1
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(env.bodies[seed-1]))
		req.Header.Set("Content-Type", "application/json")
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			b.Fatalf("iteration %d (seed %d): HTTP %d", i, seed, rec.Code)
		}
		// Empty content values are omitted by the DTO, so the presence of
		// "content":" is exactly one non-empty delta.content value.
		n := strings.Count(rec.Body.String(), `"content":"`)
		if want := env.expected[seed-1]; n != want {
			b.Fatalf("iteration %d (seed %d): handler streamed %d content deltas, direct generate produced %d tokens", i, seed, n, want)
		}
		tokens += n
	}
	reportThroughput(b, tokens)
}

// BenchmarkSSEWrite isolates the pure SSE serialization cost per token:
// json.Marshal of one chat chunk, the "data: ..." prefix, the write, and the
// flush through a discard ResponseWriter. No engine, no HTTP transport.
func BenchmarkSSEWrite(b *testing.B) {
	_ = benchGuard(b)
	w := newDiscardResponseWriter()
	sw := newSSEWriter(w, w)
	chunk := chatChunk{
		ID:      "chatcmpl-bench000000000000",
		Object:  "chat.completion.chunk",
		Created: 1700000000,
		Model:   benchModelID,
		Choices: []chatChoiceDelta{{Index: 0, Delta: delta{Content: benchTokenText}}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sw.write(chunk)
	}
}

// discardResponseWriter is an http.ResponseWriter + http.Flusher that drops the
// bytes, so the SSE benchmark measures serialization and not the sink.
type discardResponseWriter struct {
	header http.Header
}

func newDiscardResponseWriter() *discardResponseWriter {
	return &discardResponseWriter{header: make(http.Header)}
}

func (w *discardResponseWriter) Header() http.Header { return w.header }

func (w *discardResponseWriter) Write(p []byte) (int, error) { return len(p), nil }

func (w *discardResponseWriter) WriteHeader(int) {}

func (w *discardResponseWriter) Flush() {}
