// Package server implements the OpenAI-compatible provider: an HTTP surface compatible
// with the TinyCoder extension's OpenAI-style client, backed by the
// tinycoder-inference engine via the model Manager.
package server

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/mgorshkov/tinycoder-openai/internal/config"
)

// Server holds the HTTP handler and its dependencies.
type Server struct {
	cfg     config.Config
	mgr     *Manager
	log     *slog.Logger
	handler http.Handler
}

// New builds the HTTP server and routes.
func New(cfg config.Config, mgr *Manager, log *slog.Logger) *Server {
	if log == nil {
		log = slog.Default()
	}
	s := &Server{cfg: cfg, mgr: mgr, log: log}
	s.handler = s.routes()
	return s
}

// Handler returns the root http.Handler (with middleware applied).
func (s *Server) Handler() http.Handler { return s.handler }

// routes registers all endpoints under the canonical /v1 prefix (served at
// https://api.tinycoder.dev/v1), plus the operational endpoints: /healthz and
// the OpenAPI document.
func (s *Server) routes() http.Handler {
	mux := http.NewServeMux()

	// Canonical public API surface: https://api.tinycoder.dev/v1/...
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/v1/generate", s.handleGenerate)
	mux.HandleFunc("/v1/completions", s.handleCompletions)
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)

	// Operational endpoints.
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/openai.json", s.handleSpec)
	mux.HandleFunc("/openai.yaml", s.handleSpec)

	return s.withMiddleware(mux)
}

// withMiddleware applies CORS, panic recovery and request logging.
func (s *Server) withMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin != "" && s.originAllowed(origin) {
			h := w.Header()
			h.Set("Access-Control-Allow-Origin", origin)
			h.Set("Vary", "Origin")
			h.Set("Access-Control-Allow-Credentials", "true")
			h.Set("Access-Control-Allow-Headers", "Authorization, Content-Type, Accept, api-key, x-api-key")
			h.Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		}
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		defer func() {
			if err := recover(); err != nil {
				s.log.Error("panic recovered", "err", err, "path", r.URL.Path)
				if !rec.wrote {
					writeError(rec, http.StatusInternalServerError, "internal error", "server_error", "")
				}
			}
			s.log.Info("request", "method", r.Method, "path", r.URL.Path,
				"status", rec.status, "elapsed", time.Since(start).String(),
				"remote", clientIP(r))
		}()
		next.ServeHTTP(rec, r)
	})
}

func (s *Server) originAllowed(origin string) bool {
	for _, o := range s.cfg.AllowedOrigins {
		if o == "*" || strings.EqualFold(o, origin) {
			return true
		}
	}
	return false
}

// authorize enforces API-key authentication. Any key configured via api_keys is
// accepted; a missing or unknown key is rejected with 401. When no keys are
// configured the server is open.
func (s *Server) authorize(w http.ResponseWriter, r *http.Request) bool {
	keys := s.cfg.AuthKeySet()
	if len(keys) == 0 {
		return true
	}
	token := apiKeyFromRequest(r)
	if _, ok := keys[token]; token == "" || !ok {
		w.Header().Set("WWW-Authenticate", "Bearer")
		writeError(w, http.StatusUnauthorized, "invalid or missing API key", "authentication_error", "invalid_api_key")
		return false
	}
	return true
}

// apiKeyFromRequest extracts a presented API key from the request, accepting the
// header forms used by OpenAI-compatible clients (OpenAI SDK, Azure OpenAI,
// @ai-sdk/openai-compatible, Kilo/opencode, Roo Code) plus a query parameter:
//
//   - Authorization: Bearer <key>
//   - Authorization: <key>        (some clients send the bare key)
//   - api-key: <key>              (Azure OpenAI / OpenAPI security scheme)
//   - x-api-key: <key>            (Anthropic-style compatible clients / AI SDK)
//   - X-Api-Key: <key>
//   - ?api_key=<key>              (query parameter)
//
// The first non-empty match wins; an explicit Authorization header takes
// precedence so its exact value is validated even when the bare form is used.
func apiKeyFromRequest(r *http.Request) string {
	if h := strings.TrimSpace(r.Header.Get("Authorization")); h != "" {
		if len(h) >= len("Bearer ") && strings.EqualFold(h[:len("Bearer ")], "Bearer ") {
			return strings.TrimSpace(h[len("Bearer "):])
		}
		// Some OpenAI-compatible clients send the raw key with no scheme.
		return h
	}
	for _, name := range []string{"api-key", "x-api-key"} {
		if v := strings.TrimSpace(r.Header.Get(name)); v != "" {
			return v
		}
	}
	return strings.TrimSpace(r.URL.Query().Get("api_key"))
}

// requestContext returns a context bounded by the configured request timeout
// and canceled when the client disconnects.
func (s *Server) requestContext(r *http.Request) (context.Context, context.CancelFunc) {
	return context.WithTimeout(r.Context(), s.cfg.RequestTimeout())
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ok",
		"time":   time.Now().UTC().Format(time.RFC3339),
	})
}

// statusRecorder captures the response status and whether headers were written.
type statusRecorder struct {
	http.ResponseWriter
	status int
	wrote  bool
}

func (r *statusRecorder) WriteHeader(code int) {
	if !r.wrote {
		r.status = code
		r.wrote = true
	}
	r.ResponseWriter.WriteHeader(code)
}

func (r *statusRecorder) Write(b []byte) (int, error) {
	if !r.wrote {
		r.wrote = true
	}
	return r.ResponseWriter.Write(b)
}

// Flush forwards to the underlying writer when it supports http.Flusher, so
// streaming responses work.
func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func clientIP(r *http.Request) string {
	if v := r.Header.Get("X-Forwarded-For"); v != "" {
		if i := strings.IndexByte(v, ','); i >= 0 {
			return strings.TrimSpace(v[:i])
		}
		return strings.TrimSpace(v)
	}
	return r.RemoteAddr
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	_ = enc.Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg, typ, code string) {
	writeJSON(w, status, errorBody{Error: errorDetail{Message: msg, Type: typ, Code: code}})
}
