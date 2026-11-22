package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mgorshkov/tinycoder-openai/internal/config"
	"github.com/mgorshkov/tinycoder-openai/internal/engine"
)

// ErrModelNotFound is returned when a requested id is not in the registry.
var ErrModelNotFound = errors.New("model not found")

// ErrCapacity is returned when a new model cannot be loaded because the
// configured capacity is exhausted and no model could be evicted in time.
var ErrCapacity = errors.New("no capacity to load model")

// entry tracks one registry model and its (optionally loaded) engine.
type entry struct {
	id     string
	cfg    config.RegistryModel
	engine engine.Engine

	// mu serializes load/unload/generate for this model. A Generate call holds
	// it for the whole request, which is required because the underlying engine
	// uses a single KV cache and prints to stdout.
	mu sync.Mutex
	// loaded is accessed under m.mu by IsLoaded/evictIdle/residentCountLocked/
	// pickVictimLocked and under e.mu by Acquire/unload; making it atomic keeps
	// /readyz and /v1/models polling from racing an in-flight load/unload.
	loaded   atomic.Bool
	inUse    bool
	evicting bool
	lastUsed time.Time
	info     engine.ModelInfo
}

// Manager lazily loads registry models into engines, enforces a capacity on
// concurrently resident models with LRU eviction, and serializes access to each
// engine. It is safe for concurrent use.
type Manager struct {
	cfg     config.Config
	factory engine.Factory
	log     *slog.Logger

	mu      sync.Mutex
	cond    *sync.Cond
	entries map[string]*entry
	order   []string // creation order for deterministic iteration

	// sem bounds the number of in-flight generation requests across all models.
	// It is admission (queue depth) only: a caller holding a slot has not
	// necessarily started generating.
	sem chan struct{}

	// preloadMu guards the startup warm-up state read by Readiness. It is
	// independent of mu/e.mu: Readiness must stay cheap and never block behind a
	// load or a generation.
	preloadMu   sync.Mutex
	warm        map[string]bool   // models warmed at least once (sticky)
	preloadErrs map[string]string // last warm-up failure per model

	// exec serializes actual engine calls across all models. The native engine is
	// single-tenant — one process-wide ThreadPool task slot and one GPU device —
	// so two resident models must not run forward() at the same time. This is
	// what makes max_concurrent_requests > 1 safe today (the semaphore alone
	// would let two models enter Generate concurrently and race the pool); it
	// goes away with continuous batching.
	exec sync.Mutex

	stopOnce sync.Once
	stopCh   chan struct{}
}

// NewManager constructs a Manager for the given configuration and engine
// factory.
func NewManager(cfg config.Config, factory engine.Factory, log *slog.Logger) *Manager {
	if log == nil {
		log = slog.Default()
	}
	if cfg.MaxConcurrent < 1 {
		// Validate resolves 0 to one slot per model; guard direct construction
		// (tests, embedders) so a zero-capacity semaphore cannot deadlock.
		cfg.MaxConcurrent = 1
	}
	m := &Manager{
		cfg:         cfg,
		factory:     factory,
		log:         log,
		entries:     make(map[string]*entry, len(cfg.Models)),
		warm:        make(map[string]bool, len(cfg.Models)),
		preloadErrs: make(map[string]string),
		sem:         make(chan struct{}, cfg.MaxConcurrent),
		stopCh:      make(chan struct{}),
	}
	m.cond = sync.NewCond(&m.mu)
	for _, rm := range cfg.Models {
		m.entries[rm.ID] = &entry{
			id:     rm.ID,
			cfg:    rm,
			engine: factory(rm.ID),
		}
		m.order = append(m.order, rm.ID)
	}
	return m
}

// Registry returns the configured model registry.
func (m *Manager) Registry() []config.RegistryModel { return m.cfg.Models }

// IsLoaded reports whether the given model is currently resident.
func (m *Manager) IsLoaded(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	e, ok := m.entries[id]
	return ok && e.loaded.Load()
}

// ModelByID returns the registry entry for id.
func (m *Manager) ModelByID(id string) (config.RegistryModel, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	e, ok := m.entries[id]
	if !ok {
		return config.RegistryModel{}, false
	}
	return e.cfg, true
}

// Start launches the idle-eviction background loop.
func (m *Manager) Start() {
	go m.idleLoop()
}

// Stop terminates the background loop (does not unload models).
func (m *Manager) Stop() {
	m.stopOnce.Do(func() { close(m.stopCh) })
}

func (m *Manager) idleLoop() {
	interval := m.cfg.IdleTimeout() / 2
	if interval < time.Second {
		interval = time.Second
	}
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-m.stopCh:
			return
		case <-t.C:
			m.evictIdle()
		}
	}
}

func (m *Manager) evictIdle() {
	timeout := m.cfg.IdleTimeout()
	if timeout <= 0 {
		return
	}
	var victims []*entry
	m.mu.Lock()
	now := time.Now()
	for _, e := range m.entries {
		if e.loaded.Load() && !e.inUse && !e.evicting && now.Sub(e.lastUsed) > timeout {
			e.evicting = true
			victims = append(victims, e)
		}
	}
	m.mu.Unlock()
	for _, e := range victims {
		m.unload(e)
	}
}

// unload assumes the entry has been reserved (evicting=true) and unloads it.
func (m *Manager) unload(e *entry) {
	e.mu.Lock()
	err := e.engine.Unload()
	e.loaded.Store(false)
	e.info = engine.ModelInfo{}
	e.mu.Unlock()
	m.mu.Lock()
	e.evicting = false
	e.inUse = false
	m.mu.Unlock()
	if err != nil {
		m.log.Error("unload failed", "model", e.id, "err", err)
	} else {
		m.log.Info("unloaded model", "model", e.id)
	}
	m.cond.Broadcast()
}

// Lease represents exclusive access to a loaded model. Callers must call
// Release when finished.
type Lease struct {
	mgr *Manager
	e   *entry
}

// Model returns the underlying engine for the lease.
func (l *Lease) Model() engine.Engine { return l.e.engine }

// Info returns model metadata for the lease.
func (l *Lease) Info() engine.ModelInfo { return l.e.info }

// Release returns the model to the manager.
func (l *Lease) Release() {
	m := l.mgr
	e := l.e
	m.mu.Lock()
	e.inUse = false
	e.lastUsed = time.Now()
	m.mu.Unlock()
	e.mu.Unlock()
	m.cond.Broadcast()
}

// Acquire returns an exclusive lease on the loaded model for id, loading it on
// demand (evicting idle models if necessary). The context bounds the load wait.
func (m *Manager) Acquire(ctx context.Context, id string) (*Lease, error) {
	m.mu.Lock()
	e, ok := m.entries[id]
	m.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrModelNotFound, id)
	}

	// Serialize all work on this model. If another request is generating we
	// block here until it releases.
	e.mu.Lock()

	if e.loaded.Load() && e.engine.Loaded() {
		m.markInUse(e)
		m.recordWarm(e.id)
		return &Lease{mgr: m, e: e}, nil
	}
	if e.loaded.Load() {
		// Defensive: the registry thinks the model is resident but the engine
		// reports otherwise (a stale handle). Unload and fall through to a clean
		// reload so the entry can never serve against a dead engine. Under
		// normal operation this does not happen: the native engine waits for an
		// aborted generation to unwind and keeps the handle valid.
		_ = e.engine.Unload()
		e.loaded.Store(false)
		e.info = engine.ModelInfo{}
	}

	// Ensure there is room to load (evicting an idle model if at capacity).
	if err := m.ensureCapacity(ctx, e); err != nil {
		e.mu.Unlock()
		return nil, err
	}

	opts := engine.LoadOptions{
		ModelPath: m.cfg.ResolveModelPath(e.cfg),
		NThreads:  m.cfg.NThreads,
		MaxSeqLen: e.cfg.ContextLength,
	}
	// The engine gives each model a private thread pool, so a model may size it
	// independently (e.g. a small model on few threads next to a large one).
	if e.cfg.NThreads > 0 {
		opts.NThreads = e.cfg.NThreads
	}
	if opts.MaxSeqLen == 0 {
		opts.MaxSeqLen = m.cfg.MaxSeqLen
	}
	start := time.Now()
	m.log.Info("loading model", "model", e.id, "path", opts.ModelPath)
	if err := e.engine.Load(ctx, opts); err != nil {
		e.mu.Unlock()
		return nil, fmt.Errorf("load model %s: %w", e.id, err)
	}
	e.loaded.Store(true)
	m.recordWarm(e.id)
	e.info = e.engine.Info()
	if e.info.ID == "" {
		e.info.ID = e.id
	}
	if e.info.Name == "" {
		e.info.Name = e.cfg.Name
	}
	if e.info.ContextLength == 0 {
		e.info.ContextLength = opts.MaxSeqLen
	}
	if e.info.SizeBytes == 0 {
		e.info.SizeBytes = e.cfg.SizeBytes
	}
	m.log.Info("model loaded", "model", e.id, "architecture", e.info.Architecture,
		"elapsed", time.Since(start).String())

	m.markInUse(e)
	return &Lease{mgr: m, e: e}, nil
}

func (m *Manager) markInUse(e *entry) {
	m.mu.Lock()
	e.inUse = true
	e.lastUsed = time.Now()
	m.mu.Unlock()
}

// ensureCapacity guarantees that loading e will not exceed MaxLoadedModels.
// It must be called with e.mu held (e is the model being loaded) and evicts
// idle models as needed, waiting if all resident models are busy.
func (m *Manager) ensureCapacity(ctx context.Context, e *entry) error {
	deadline := time.Now().Add(m.cfg.RequestTimeout())
	for {
		m.mu.Lock()
		if m.residentCountLocked() < m.cfg.MaxLoadedModels {
			m.mu.Unlock()
			return nil
		}
		victim := m.pickVictimLocked(e)
		if victim != nil {
			// Reserve so no other goroutine targets it before we unload.
			victim.evicting = true
			m.mu.Unlock()
			m.log.Info("evicting model for capacity", "victim", victim.id, "for", e.id)
			m.unload(victim)
			return nil
		}
		// No idle victim: wait for one to free up. Re-check periodically so we
		// can honor the context deadline even if no broadcast arrives.
		if time.Now().After(deadline) {
			m.mu.Unlock()
			return fmt.Errorf("%w: %s", ErrCapacity, e.id)
		}
		m.mu.Unlock()
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(100 * time.Millisecond):
		}
	}
}

func (m *Manager) residentCountLocked() int {
	n := 0
	for _, e := range m.entries {
		if e.loaded.Load() {
			n++
		}
	}
	return n
}

// pickVictimLocked returns the least-recently-used idle loaded entry (never e).
func (m *Manager) pickVictimLocked(e *entry) *entry {
	var victim *entry
	for _, cand := range m.entries {
		if cand == e || !cand.loaded.Load() || cand.inUse || cand.evicting {
			continue
		}
		if victim == nil || cand.lastUsed.Before(victim.lastUsed) {
			victim = cand
		}
	}
	return victim
}

// acquireSem blocks until a global request slot is free or ctx is done.
func (m *Manager) acquireSem(ctx context.Context) error {
	select {
	case m.sem <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (m *Manager) releaseSem() { <-m.sem }

// PromptTokens returns the request's prompt token count and the model's
// effective context window, loading the model if needed. A zero window means
// unknown and callers must skip the check. Errors mean "cannot determine".
func (m *Manager) PromptTokens(ctx context.Context, id string, req engine.GenerateRequest) (promptTokens, window int, err error) {
	// The preflight only needs the model, not a generation slot: do not take the
	// global semaphore (that would queue the request twice).
	lease, err := m.Acquire(ctx, id)
	if err != nil {
		return 0, 0, err
	}
	defer lease.Release()

	window = m.effectiveWindow(id, lease.Info())

	promptTokens, err = lease.Model().CountPromptTokens(req)
	if err != nil {
		return 0, window, err
	}
	return promptTokens, window, nil
}

// TryPromptTokens is the non-blocking variant of PromptTokens used by the
// streaming preflight. It never waits for a busy model and never loads one: it
// only takes the entry lock when it is free and reports ok=false when the model
// is not resident, another request holds the lock (generating, loading or
// unloading), or the prompt cannot be counted. A false result therefore means
// "skip the check", which also closes the warm-then-evict race: a model that
// leaves residency between IsLoaded and here is simply not preflighted.
func (m *Manager) TryPromptTokens(id string, req engine.GenerateRequest) (promptTokens, window int, ok bool) {
	m.mu.Lock()
	e, found := m.entries[id]
	m.mu.Unlock()
	if !found {
		return 0, 0, false
	}
	if !e.mu.TryLock() {
		return 0, 0, false
	}
	defer e.mu.Unlock()
	if !e.loaded.Load() {
		return 0, 0, false
	}

	window = m.effectiveWindow(id, e.info)
	promptTokens, err := e.engine.CountPromptTokens(req)
	if err != nil || promptTokens <= 0 {
		return 0, window, false
	}
	return promptTokens, window, true
}

// effectiveWindow resolves the model's context window: the loaded engine's
// reported length, then the registry entry, then the global max seq len.
func (m *Manager) effectiveWindow(id string, info engine.ModelInfo) int {
	window := info.ContextLength
	if window == 0 {
		if rm, ok := m.ModelByID(id); ok {
			window = rm.ContextLength
		}
	}
	if window == 0 {
		window = m.cfg.MaxSeqLen
	}
	return window
}

// Generate acquires the model, runs generation, and releases it. It also
// enforces the global concurrency limit.
//
// The concurrency slot (sem) and the per-entry lock are released by the deferred
// calls above/below. This is safe even when the engine overruns: the manager's
// own per-request watchdog owns the inactivity bound (the native engine's
// no-token guard is a backstop), and the native engine bounds each generation
// call with the hard timeout, then waits for the (cooperatively aborted) engine
// call to unwind before returning, so the model lock is never released while
// engine code is still running.
func (m *Manager) Generate(ctx context.Context, id string, req engine.GenerateRequest, onToken engine.TokenFunc) (engine.Usage, error) {
	if err := m.acquireSem(ctx); err != nil {
		return engine.Usage{}, err
	}
	defer m.releaseSem()

	lease, err := m.Acquire(ctx, id)
	if err != nil {
		return engine.Usage{}, err
	}
	defer lease.Release()

	req.Model = id
	if req.NoTokenTimeout <= 0 {
		req.NoTokenTimeout = m.cfg.NoTokenTimeout()
	}
	if req.HardTimeout <= 0 {
		req.HardTimeout = m.cfg.HardRequestTimeout()
	}

	m.exec.Lock()
	defer m.exec.Unlock()

	if req.NoTokenTimeout <= 0 {
		return lease.Model().Generate(ctx, req, onToken)
	}

	// Per-request inactivity watchdog. It cancels the generation when neither a
	// token nor an engine prefill-progress event arrives within NoTokenTimeout,
	// and is started only after Acquire, so a cold model load (which can take
	// minutes) never counts as inactivity.
	var last atomic.Int64
	last.Store(time.Now().UnixNano())
	prevProgress := req.OnProgress
	// Tokens and engine progress both refresh the inactivity clock, but the
	// caller's OnProgress is chained only into the progress path: it means
	// "progress that has not produced a token yet" and must not fire per token.
	req.OnProgress = func() {
		last.Store(time.Now().UnixNano())
		if prevProgress != nil {
			prevProgress()
		}
	}
	wrapped := func(t engine.Token) bool {
		last.Store(time.Now().UnixNano())
		if onToken == nil {
			return true
		}
		return onToken(t)
	}
	onToken = wrapped
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var tripped atomic.Bool
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		iv := req.NoTokenTimeout / 4
		if iv < 5*time.Millisecond {
			iv = 5 * time.Millisecond
		}
		if iv > time.Second {
			iv = time.Second
		}
		tick := time.NewTicker(iv)
		defer tick.Stop()
		for {
			select {
			case <-stop:
				return
			case <-tick.C:
				if time.Since(time.Unix(0, last.Load())) > req.NoTokenTimeout {
					tripped.Store(true)
					cancel()
					return
				}
			}
		}
	}()
	usage, err := lease.Model().Generate(ctx, req, onToken)
	if tripped.Load() {
		return usage, engine.ErrNoToken
	}
	return usage, err
}
