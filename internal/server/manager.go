package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
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
	mu       sync.Mutex
	loaded   bool
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
	sem chan struct{}

	stopOnce sync.Once
	stopCh   chan struct{}
}

// NewManager constructs a Manager for the given configuration and engine
// factory.
func NewManager(cfg config.Config, factory engine.Factory, log *slog.Logger) *Manager {
	if log == nil {
		log = slog.Default()
	}
	m := &Manager{
		cfg:     cfg,
		factory: factory,
		log:     log,
		entries: make(map[string]*entry, len(cfg.Models)),
		sem:     make(chan struct{}, cfg.MaxConcurrent),
		stopCh:  make(chan struct{}),
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
	return ok && e.loaded
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
		if e.loaded && !e.inUse && !e.evicting && now.Sub(e.lastUsed) > timeout {
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
	e.loaded = false
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

	if e.loaded {
		m.markInUse(e)
		return &Lease{mgr: m, e: e}, nil
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
	if opts.MaxSeqLen == 0 {
		opts.MaxSeqLen = m.cfg.MaxSeqLen
	}
	start := time.Now()
	m.log.Info("loading model", "model", e.id, "path", opts.ModelPath)
	if err := e.engine.Load(ctx, opts); err != nil {
		e.mu.Unlock()
		return nil, fmt.Errorf("load model %s: %w", e.id, err)
	}
	e.loaded = true
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
		if e.loaded {
			n++
		}
	}
	return n
}

// pickVictimLocked returns the least-recently-used idle loaded entry (never e).
func (m *Manager) pickVictimLocked(e *entry) *entry {
	var victim *entry
	for _, cand := range m.entries {
		if cand == e || !cand.loaded || cand.inUse || cand.evicting {
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

// Generate acquires the model, runs generation, and releases it. It also
// enforces the global concurrency limit.
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
	return lease.Model().Generate(ctx, req, onToken)
}
