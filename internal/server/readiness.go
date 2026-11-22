package server

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"
)

// ModelReadiness is one model's warm-up state in the GET /readyz response.
//
// Loaded is sticky on success: it stays true once the model has been warmed at
// least once, even if an idle eviction later unloads it. The process stays
// healthy and the model remains loadable, so a warmed process never flips back
// to "not ready" merely because its model went idle.
type ModelReadiness struct {
	ID     string `json:"id"`
	Loaded bool   `json:"loaded"`
	Error  string `json:"error,omitempty"`
}

// readinessResponse is the GET /readyz body.
type readinessResponse struct {
	Status string           `json:"status"`
	Models []ModelReadiness `json:"models"`
}

// StartPreload warms the configured models in the background so an orchestrator
// can hold traffic until /readyz turns 200. It returns immediately; readiness
// reports progress. It is a no-op when no model is configured for preload.
func (s *Server) StartPreload() {
	ids := s.cfg.PreloadIDs()
	if len(ids) == 0 {
		return
	}
	s.log.Info("preloading models", "models", ids)
	go s.mgr.Preload(context.Background(), ids)
}

// handleReady reports whether the configured preload targets are warm. It is
// unauthenticated like /healthz: an orchestrator needs it before it has a key
// (or a route) and it exposes no model data.
func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error", "")
		return
	}
	status, models := s.mgr.Readiness(s.cfg.PreloadIDs())
	code := http.StatusOK
	if status != "ready" {
		code = http.StatusServiceUnavailable
	}
	writeJSON(w, code, readinessResponse{Status: status, Models: models})
}

// Preload warms ids sequentially, each bounded by preload_timeout_sec when
// non-zero. Sequential loads are deliberate: loading several models
// concurrently would thrash max_loaded_models. A failed target is recorded for
// Readiness (and logged); there is no retry loop, but any later successful load
// of that model through Acquire clears the error.
func (m *Manager) Preload(ctx context.Context, ids []string) {
	for _, id := range ids {
		if err := m.preloadOne(ctx, id); err != nil {
			m.log.Error("preload failed", "model", id, "err", err)
			m.recordPreloadError(id, err)
			continue
		}
		m.log.Info("preloaded model", "model", id)
	}
}

// preloadOne performs one bounded Acquire/Release warm-up. Acquire hands the
// deadline to Engine.Load, so the budget is enforced only by engines that honor
// context cancellation; the synchronous native load cannot be interrupted and
// may outlast it.
func (m *Manager) preloadOne(ctx context.Context, id string) error {
	timeout := time.Duration(m.cfg.PreloadTimeoutSec) * time.Second
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}
	lease, err := m.Acquire(ctx, id)
	if err != nil {
		if timeout > 0 && errors.Is(err, context.DeadlineExceeded) {
			return fmt.Errorf("warm-up timed out after %s", timeout)
		}
		return err
	}
	lease.Release()
	return nil
}

// Readiness reports the preload status ("ready", "loading" or "error") and one
// entry per id. With no ids there is nothing to wait for: a lazily loaded
// server is ready immediately.
func (m *Manager) Readiness(ids []string) (string, []ModelReadiness) {
	models := make([]ModelReadiness, 0, len(ids))
	if len(ids) == 0 {
		return "ready", models
	}
	m.preloadMu.Lock()
	defer m.preloadMu.Unlock()
	allLoaded := true
	anyErr := false
	for _, id := range ids {
		mr := ModelReadiness{ID: id, Loaded: m.warm[id]}
		if msg, ok := m.preloadErrs[id]; ok && !mr.Loaded {
			mr.Error = msg
			anyErr = true
		}
		if !mr.Loaded {
			allLoaded = false
		}
		models = append(models, mr)
	}
	switch {
	case allLoaded:
		return "ready", models
	case anyErr:
		// A failed warm-up is actionable now; report it even while another
		// target is still loading.
		return "error", models
	default:
		return "loading", models
	}
}

// recordWarm marks a model as successfully warmed and clears any previous
// warm-up error. It is called on every successful Acquire, so a lazy request
// (or a retried warm-up) repairs readiness after a failed preload.
func (m *Manager) recordWarm(id string) {
	m.preloadMu.Lock()
	m.warm[id] = true
	delete(m.preloadErrs, id)
	m.preloadMu.Unlock()
}

// recordPreloadError stores a warm-up failure for Readiness.
func (m *Manager) recordPreloadError(id string, err error) {
	m.preloadMu.Lock()
	m.preloadErrs[id] = err.Error()
	m.preloadMu.Unlock()
}
