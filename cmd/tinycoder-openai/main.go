// Command tinycoder-openai serves the TinyCoder inference engine over an
// OpenAI-compatible HTTP API so users can select https://api.tinycoder.dev/v1
// (or any deployment URL), list models from /v1/models, and generate text.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/mgorshkov/tinycoder-openai/internal/config"
	"github.com/mgorshkov/tinycoder-openai/internal/server"
	"github.com/mgorshkov/tinycoder-openai/internal/wiring"
)

// version is overridable at build time with -ldflags "-X main.version=...".
var version = "dev"

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "fatal:", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		configPath = flag.String("config", envOr("TINYCODER_CONFIG", ""), "path to JSON config file")
		listen     = flag.String("listen", "", "listen address (overrides config), e.g. :8080")
		modelsDir  = flag.String("models-dir", "", "directory containing GGUF models (overrides config)")
		logLevel   = flag.String("log-level", "", "log level: debug|info|warn|error (overrides config)")
		model      = flag.String("model", "", "one-model mode: model file path (replaces the config models list)")
		modelID    = flag.String("model-id", "", "one-model mode: served model id (defaults to the slugified base name of the file without a .gguf extension)")
		ctxLen     = flag.Int("ctx", 0, "one-model mode: context length for --model (0 inherits max_seq_len, 8192 by default)")
		showVer    = flag.Bool("version", false, "print version and exit")
	)
	flag.Parse()

	if *showVer {
		fmt.Printf("tinycoder-openai %s (native engine: %v)\n", version, wiring.Native)
		return nil
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		return err
	}
	if err := cfg.ApplyEnv(); err != nil {
		return err
	}
	if *listen != "" {
		cfg.Listen = *listen
	}
	if *modelsDir != "" {
		cfg.ModelsDir = *modelsDir
	}
	if *logLevel != "" {
		cfg.LogLevel = *logLevel
	}
	// One-model-per-process mode (like llama-server). These flags win over the
	// config file's registry: --model replaces the whole "models" list.
	if err := cfg.ApplyModelFlags(*model, *modelID, *ctxLen); err != nil {
		return err
	}
	// The config file is the authoritative registry: /v1/models advertises only
	// the models declared under "models", which replace the built-in list. Models
	// found by scanning models_dir are intentionally NOT advertised, so an
	// untested or unsupported GGUF on disk can never be exposed (or selected)
	// through the API. Add it to "models" to publish it.
	if err := cfg.Validate(); err != nil {
		return err
	}

	log := newLogger(cfg.LogLevel)
	slog.SetDefault(log)

	if !wiring.Native {
		log.Warn("native inference engine not compiled in; serving the stub engine",
			"hint", "this binary was built with the tinycoder_stub tag; build the native engine with CGO_ENABLED=1 (see scripts/build.sh)")
	}
	log.Info("starting tinycoder-openai", "version", version, "listen", cfg.Listen,
		"models", len(cfg.Models), "native", wiring.Native)

	mgr := server.NewManager(cfg, wiring.NewFactory(), log)
	mgr.Start()
	defer mgr.Stop()

	srv := server.New(cfg, mgr, log)
	// Warm the configured models in the background. This must not block startup:
	// the listener comes up first and /readyz reports the warm-up progress.
	srv.StartPreload()

	httpSrv := &http.Server{
		Addr:              cfg.Listen,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 15 * time.Second,
		// No ReadTimeout: request bodies are small but streaming responses can
		// be long-lived. IdleTimeout bounds keep-alive connections.
		IdleTimeout: 120 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		log.Info("shutdown signal received, draining connections")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		return httpSrv.Shutdown(shutdownCtx)
	}
}

// newLogger builds a slog.Logger at the requested level.
func newLogger(level string) *slog.Logger {
	var lvl slog.Level
	switch strings.ToLower(level) {
	case "debug":
		lvl = slog.LevelDebug
	case "warn", "warning":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}
	return slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: lvl}))
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
