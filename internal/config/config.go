// Package config loads tinycoder-openai server configuration from a JSON file,
// environment variables and command-line flags (in increasing precedence).
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// RegistryModel describes one model exposed by /v1/models.
type RegistryModel struct {
	ID            string `json:"id"`
	Name          string `json:"name"`
	File          string `json:"file"`
	Architecture  string `json:"architecture,omitempty"`
	Quantization  string `json:"quantization,omitempty"`
	ContextLength int    `json:"context_length,omitempty"`
	SizeBytes     int64  `json:"size_bytes,omitempty"`
	Description   string `json:"description,omitempty"`
}

// Config is the fully resolved server configuration.
type Config struct {
	Listen            string          `json:"listen"`
	PublicURL         string          `json:"public_url"`
	AuthToken         string          `json:"auth_token"`
	APIKeys           []string        `json:"api_keys"`
	ToolCalling       bool            `json:"tool_calling"`
	AllowedOrigins    []string        `json:"allowed_origins"`
	ModelsDir         string          `json:"models_dir"`
	ScanModelsDir     bool            `json:"scan_models_dir"`
	MaxLoadedModels   int             `json:"max_loaded_models"`
	IdleTimeoutSec    int             `json:"idle_timeout_sec"`
	NThreads          int             `json:"n_threads"`
	MaxSeqLen         int             `json:"max_seq_len"`
	DefaultMaxTokens  int             `json:"default_max_tokens"`
	MaxPromptTokens   int             `json:"max_prompt_tokens"`
	RequestTimeoutSec int             `json:"request_timeout_sec"`
	MaxConcurrent     int             `json:"max_concurrent_requests"`
	LogLevel          string          `json:"log_level"`
	Models            []RegistryModel `json:"models"`
}

// Default returns a configuration with sensible production defaults and a
// built-in model registry covering the model families supported by
// tinycoder-inference (see its README).
func Default() Config {
	return Config{
		Listen:    ":8080",
		PublicURL: "https://api.tinycoder.dev/v1",
		AuthToken: "",
		// Tool calling is a text-based protocol injected into the chat prompt;
		// see internal/server/tools.go. On by default.
		ToolCalling:    true,
		AllowedOrigins: []string{"*"},
		ModelsDir:      "/data/models",
		// Only models declared in "models" are advertised by /v1/models. Model
		// directory scanning is opt-in (TINYCODER_SCAN_MODELS_DIR / DiscoverModels)
		// for tooling that wants to enumerate on-disk GGUFs; it is never merged
		// into the served registry.
		ScanModelsDir:     false,
		MaxLoadedModels:   1,
		IdleTimeoutSec:    900,
		NThreads:          0,
		MaxSeqLen:         8192,
		DefaultMaxTokens:  2048,
		MaxPromptTokens:   32768,
		RequestTimeoutSec: 600,
		MaxConcurrent:     1,
		LogLevel:          "info",
		Models:            builtinModels(),
	}
}

func builtinModels() []RegistryModel {
	return []RegistryModel{
		{
			ID:            "qwen3.6-35b-a3b",
			Name:          "Qwen3.6-35B-A3B",
			File:          "qwen/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
			Architecture:  "qwen35moe",
			Quantization:  "UD-Q4_K_M",
			ContextLength: 8192,
			Description:   "35B-A3B mixture-of-experts coding model (recommended).",
		},
		{
			ID:            "qwen3.8-27b",
			Name:          "Qwen3.8-27B",
			File:          "qwen/Qwen3.8-27B-UD-Q4_K_M.gguf",
			Architecture:  "qwen35",
			Quantization:  "UD-Q4_K_M",
			ContextLength: 8192,
			Description:   "27B dense hybrid (gated-delta-net + full attention).",
		},
		{
			ID:            "qwen2.5-coder-1.5b",
			Name:          "Qwen2.5-Coder-1.5B",
			File:          "qwen/Qwen2.5-Coder-1.5B-Instruct-Q2_K.gguf",
			Architecture:  "qwen2",
			Quantization:  "Q2_K",
			ContextLength: 4096,
			Description:   "Lightweight fast coding model.",
		},
		{
			ID:            "qwen2.5-coder-7b",
			Name:          "Qwen2.5-Coder-7B",
			File:          "qwen/Qwen2.5-Coder-7B-Instruct-IQ3_XXS.gguf",
			Architecture:  "qwen2",
			Quantization:  "IQ3_XXS",
			ContextLength: 4096,
			Description:   "Recommended balanced 7B coding model.",
		},
		{
			ID:            "gemma4-26b-a4b",
			Name:          "Gemma 4 26B-A4B IT",
			File:          "gemma/gemma-4-26b-a4b-it-Q4_K_XL.gguf",
			Architecture:  "gemma4",
			Quantization:  "Q4_K_XL",
			ContextLength: 8192,
			Description:   "Instruction-tuned Gemma MoE.",
		},
	}
}

// Load reads a JSON config file and overlays it on the defaults. A missing
// path is not an error (defaults are returned unchanged).
func Load(path string) (Config, error) {
	cfg := Default()
	if path == "" {
		return cfg, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return cfg, fmt.Errorf("read config %q: %w", path, err)
	}
	// Decode into an overlay struct so absent keys keep defaults.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return cfg, fmt.Errorf("parse config %q: %w", path, err)
	}
	// Re-marshal merged: decode defaults then the file over the top.
	merged, err := json.Marshal(cfg)
	if err != nil {
		return cfg, err
	}
	var base map[string]json.RawMessage
	if err := json.Unmarshal(merged, &base); err != nil {
		return cfg, err
	}
	for k, v := range raw {
		base[k] = v
	}
	final, err := json.Marshal(base)
	if err != nil {
		return cfg, err
	}
	if err := json.Unmarshal(final, &cfg); err != nil {
		return cfg, fmt.Errorf("apply config %q: %w", path, err)
	}
	return cfg, nil
}

// ApplyEnv overlays environment variables (TINYCODER_*).
func (c *Config) ApplyEnv() error {
	if v := os.Getenv("TINYCODER_LISTEN"); v != "" {
		c.Listen = v
	}
	if v := os.Getenv("TINYCODER_PUBLIC_URL"); v != "" {
		c.PublicURL = v
	}
	if v := os.Getenv("TINYCODER_AUTH_TOKEN"); v != "" {
		c.AuthToken = v
	}
	if v := os.Getenv("TINYCODER_API_KEYS"); v != "" {
		c.APIKeys = splitCSV(v)
	}
	if v := os.Getenv("TINYCODER_MODELS_DIR"); v != "" {
		c.ModelsDir = v
	}
	if v := os.Getenv("TINYCODER_ALLOWED_ORIGINS"); v != "" {
		c.AllowedOrigins = splitCSV(v)
	}
	if v := os.Getenv("TINYCODER_LOG_LEVEL"); v != "" {
		c.LogLevel = v
	}
	if err := envInt("TINYCODER_MAX_LOADED_MODELS", &c.MaxLoadedModels); err != nil {
		return err
	}
	if err := envInt("TINYCODER_IDLE_TIMEOUT_SEC", &c.IdleTimeoutSec); err != nil {
		return err
	}
	if err := envInt("TINYCODER_THREADS", &c.NThreads); err != nil {
		return err
	}
	if err := envInt("TINYCODER_MAX_SEQ_LEN", &c.MaxSeqLen); err != nil {
		return err
	}
	if err := envInt("TINYCODER_DEFAULT_MAX_TOKENS", &c.DefaultMaxTokens); err != nil {
		return err
	}
	if err := envInt("TINYCODER_MAX_PROMPT_TOKENS", &c.MaxPromptTokens); err != nil {
		return err
	}
	if err := envInt("TINYCODER_REQUEST_TIMEOUT_SEC", &c.RequestTimeoutSec); err != nil {
		return err
	}
	if err := envInt("TINYCODER_MAX_CONCURRENT", &c.MaxConcurrent); err != nil {
		return err
	}
	if err := envBool("TINYCODER_SCAN_MODELS_DIR", &c.ScanModelsDir); err != nil {
		return err
	}
	if err := envBool("TINYCODER_TOOL_CALLING", &c.ToolCalling); err != nil {
		return err
	}
	return nil
}

// Validate checks the resolved configuration for consistency.
func (c *Config) Validate() error {
	if c.Listen == "" {
		return fmt.Errorf("listen address must not be empty")
	}
	if c.MaxLoadedModels < 1 {
		return fmt.Errorf("max_loaded_models must be >= 1")
	}
	if c.MaxConcurrent < 1 {
		return fmt.Errorf("max_concurrent_requests must be >= 1")
	}
	if c.DefaultMaxTokens < 1 {
		return fmt.Errorf("default_max_tokens must be >= 1")
	}
	seen := map[string]bool{}
	for _, m := range c.Models {
		if m.ID == "" {
			return fmt.Errorf("model entry with empty id")
		}
		if seen[m.ID] {
			return fmt.Errorf("duplicate model id %q", m.ID)
		}
		seen[m.ID] = true
	}
	return nil
}

// AuthKeySet returns the set of accepted API keys: the union of the legacy
// single auth_token and the api_keys list, with blank entries ignored. An empty
// set means authentication is disabled (the server is open).
func (c Config) AuthKeySet() map[string]struct{} {
	set := map[string]struct{}{}
	if k := strings.TrimSpace(c.AuthToken); k != "" {
		set[k] = struct{}{}
	}
	for _, k := range c.APIKeys {
		if k = strings.TrimSpace(k); k != "" {
			set[k] = struct{}{}
		}
	}
	return set
}

// IdleTimeout returns the configured idle-eviction window.
func (c Config) IdleTimeout() time.Duration {
	return time.Duration(c.IdleTimeoutSec) * time.Second
}

// RequestTimeout returns the per-request generation deadline.
func (c Config) RequestTimeout() time.Duration {
	return time.Duration(c.RequestTimeoutSec) * time.Second
}

// ResolveModelPath returns the absolute GGUF path for a registry entry. A file
// field that is already absolute is returned as-is; otherwise it is joined to
// ModelsDir.
func (c Config) ResolveModelPath(m RegistryModel) string {
	if filepath.IsAbs(m.File) {
		return m.File
	}
	return filepath.Join(c.ModelsDir, m.File)
}

// DiscoverModels walks ModelsDir for *.gguf files and returns registry entries
// for any that are not already covered by an existing entry's resolved path.
// IDs are derived from the file name (lower-cased, non-alphanumerics replaced
// with '-'). Discovery is best-effort: unreadable directories yield no results.
func (c Config) DiscoverModels() []RegistryModel {
	if !c.ScanModelsDir || c.ModelsDir == "" {
		return nil
	}
	existing := map[string]bool{}
	for _, m := range c.Models {
		existing[c.ResolveModelPath(m)] = true
	}
	var found []RegistryModel
	_ = filepath.WalkDir(c.ModelsDir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		if !strings.EqualFold(filepath.Ext(path), ".gguf") {
			return nil
		}
		if existing[path] {
			return nil
		}
		rel, relErr := filepath.Rel(c.ModelsDir, path)
		if relErr != nil {
			rel = path
		}
		var size int64
		if info, statErr := d.Info(); statErr == nil {
			size = info.Size()
		}
		base := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
		found = append(found, RegistryModel{
			ID:        slugify(base),
			Name:      base,
			File:      rel,
			SizeBytes: size,
		})
		existing[path] = true
		return nil
	})
	return found
}

// slugify converts a model file base name into a URL-safe id.
func slugify(s string) string {
	var b strings.Builder
	prevDash := false
	for _, r := range strings.ToLower(s) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '.':
			b.WriteRune(r)
			prevDash = false
		default:
			if !prevDash {
				b.WriteByte('-')
				prevDash = true
			}
		}
	}
	return strings.Trim(b.String(), "-")
}

func envInt(key string, dst *int) error {
	v := os.Getenv(key)
	if v == "" {
		return nil
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fmt.Errorf("%s: %w", key, err)
	}
	*dst = n
	return nil
}

func envBool(key string, dst *bool) error {
	v := os.Getenv(key)
	if v == "" {
		return nil
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		return fmt.Errorf("%s: %w", key, err)
	}
	*dst = b
	return nil
}

func splitCSV(v string) []string {
	parts := strings.Split(v, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
