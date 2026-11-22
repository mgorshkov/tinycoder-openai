package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDiscoverModels(t *testing.T) {
	dir := t.TempDir()
	// Create a nested GGUF file and a non-GGUF file.
	if err := os.MkdirAll(filepath.Join(dir, "qwen"), 0o755); err != nil {
		t.Fatal(err)
	}
	gguf := filepath.Join(dir, "qwen", "My Model-v2.Q4_K_M.gguf")
	if err := os.WriteFile(gguf, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg := Default()
	cfg.ModelsDir = dir
	cfg.ScanModelsDir = true
	// Existing entry points at a DIFFERENT file, so discovery should still find
	// the one real .gguf on disk.
	cfg.Models = []RegistryModel{{ID: "custom", Name: "Custom", File: filepath.Join(dir, "other.gguf")}}

	got := cfg.DiscoverModels()
	if len(got) != 1 {
		t.Fatalf("expected 1 discovered model, got %d: %+v", len(got), got)
	}
	if got[0].ID == "" || got[0].File == "" {
		t.Fatalf("incomplete entry: %+v", got[0])
	}
	if got[0].SizeBytes != 1 {
		t.Fatalf("expected size 1, got %d", got[0].SizeBytes)
	}
}

func TestDiscoverModelsDeduplicates(t *testing.T) {
	dir := t.TempDir()
	gguf := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(gguf, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := Default()
	cfg.ModelsDir = dir
	cfg.ScanModelsDir = true
	// Existing entry already covers this exact file (absolute path match).
	cfg.Models = []RegistryModel{{ID: "m", File: gguf}}
	if got := cfg.DiscoverModels(); len(got) != 0 {
		t.Fatalf("expected dedup to yield 0, got %+v", got)
	}
}

func TestLoadOverlaysDefaults(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.json")
	if err := os.WriteFile(path, []byte(`{"listen":"127.0.0.1:9999","max_loaded_models":3}`), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Listen != "127.0.0.1:9999" {
		t.Fatalf("listen not overlaid: %q", cfg.Listen)
	}
	if cfg.MaxLoadedModels != 3 {
		t.Fatalf("max_loaded_models not overlaid: %d", cfg.MaxLoadedModels)
	}
	// Unspecified fields keep defaults (including the built-in registry).
	if cfg.DefaultMaxTokens != Default().DefaultMaxTokens {
		t.Fatal("default_max_tokens should retain the default")
	}
	if len(cfg.Models) == 0 {
		t.Fatal("built-in registry should be retained")
	}
}

// TestLoadReplacesModels verifies that a "models" list present in the file
// replaces the built-in registry exactly: fields the file omits must be zero,
// never inherited from the built-in model at the same slice index (report
// defect 5: a one-model config leaked architecture/quantization/description
// from built-in model 0).
func TestLoadReplacesModels(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.json")
	body := `{"models":[{"id":"ornith","name":"Ornith","file":"ornith.gguf","context_length":33536}]}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Models) != 1 {
		t.Fatalf("len(models) = %d, want 1: %+v", len(cfg.Models), cfg.Models)
	}
	m := cfg.Models[0]
	if m.ID != "ornith" {
		t.Fatalf("id = %q, want %q", m.ID, "ornith")
	}
	if m.ContextLength != 33536 {
		t.Fatalf("context_length = %d, want 33536", m.ContextLength)
	}
	if m.Architecture != "" || m.Quantization != "" || m.Description != "" {
		t.Fatalf("file-omitted fields leaked from the built-in registry: architecture=%q quantization=%q description=%q",
			m.Architecture, m.Quantization, m.Description)
	}
	// Every other default must survive the fresh-zero decode.
	def := Default()
	if cfg.Listen != def.Listen {
		t.Fatalf("listen = %q, want default %q", cfg.Listen, def.Listen)
	}
	if cfg.DefaultMaxTokens != def.DefaultMaxTokens {
		t.Fatalf("default_max_tokens = %d, want default %d",
			cfg.DefaultMaxTokens, def.DefaultMaxTokens)
	}
}

// TestLoadNullValuesKeepDefaults verifies that an explicit JSON null for a
// top-level key keeps that field's default instead of zeroing it, so an
// existing config.json that uses null keeps starting and behaving as before.
func TestLoadNullValuesKeepDefaults(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.json")
	body := `{"listen":null,"allowed_origins":null,"tool_calling":null,"max_seq_len":null,"models":null}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	def := Default()
	if cfg.Listen != def.Listen {
		t.Fatalf("listen = %q, want default %q", cfg.Listen, def.Listen)
	}
	if cfg.ToolCalling != def.ToolCalling {
		t.Fatalf("tool_calling = %v, want default %v", cfg.ToolCalling, def.ToolCalling)
	}
	if cfg.MaxSeqLen != def.MaxSeqLen {
		t.Fatalf("max_seq_len = %d, want default %d", cfg.MaxSeqLen, def.MaxSeqLen)
	}
	if len(cfg.AllowedOrigins) != len(def.AllowedOrigins) {
		t.Fatalf("allowed_origins = %v, want default %v", cfg.AllowedOrigins, def.AllowedOrigins)
	}
	for i := range def.AllowedOrigins {
		if cfg.AllowedOrigins[i] != def.AllowedOrigins[i] {
			t.Fatalf("allowed_origins = %v, want default %v", cfg.AllowedOrigins, def.AllowedOrigins)
		}
	}
	if len(cfg.Models) != len(def.Models) {
		t.Fatalf("len(models) = %d, want default %d", len(cfg.Models), len(def.Models))
	}
	for i := range def.Models {
		if cfg.Models[i].ID != def.Models[i].ID {
			t.Fatalf("models[%d].id = %q, want default %q", i, cfg.Models[i].ID, def.Models[i].ID)
		}
	}
}

// TestLoadDuplicateCaseVariantKeysDeterministic verifies that a file containing
// several case variants of one top-level key resolves deterministically: the
// exact canonical spelling ("models") wins, not whichever key Go's map
// iteration happens to visit last.
func TestLoadDuplicateCaseVariantKeysDeterministic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.json")
	body := `{"models":[{"id":"canonical","name":"Canonical","file":"canonical.gguf"}],` +
		`"Models":[{"id":"variant","name":"Variant","file":"variant.gguf"}]}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 50; i++ {
		cfg, err := Load(path)
		if err != nil {
			t.Fatal(err)
		}
		if len(cfg.Models) != 1 || cfg.Models[0].ID != "canonical" {
			t.Fatalf("iteration %d: models = %+v, want exactly the canonical \"models\" entry",
				i, cfg.Models)
		}
	}
}

// TestLoadEmptyModelsServesNone verifies an explicit empty list serves no
// models rather than falling back to the built-ins.
func TestLoadEmptyModelsServesNone(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.json")
	if err := os.WriteFile(path, []byte(`{"models":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Models) != 0 {
		t.Fatalf("len(models) = %d, want 0: %+v", len(cfg.Models), cfg.Models)
	}
}

// TestLoadModelsKeyCaseInsensitive verifies a "Models"/"MODELS" top-level key
// behaves like "models" instead of adding a second key that json.Marshal sorts
// after the built-in one (which silently ignored the file's list).
func TestLoadModelsKeyCaseInsensitive(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.json")
	body := `{"Models":[{"id":"solo","name":"Solo","file":"solo.gguf"}]}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Models) != 1 {
		t.Fatalf("len(models) = %d, want 1: %+v", len(cfg.Models), cfg.Models)
	}
	if cfg.Models[0].ID != "solo" {
		t.Fatalf("id = %q, want %q", cfg.Models[0].ID, "solo")
	}
}

// TestLoadMissingModelsKeepsBuiltin verifies a config file without a "models"
// key still serves the built-in registry.
func TestLoadMissingModelsKeepsBuiltin(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.json")
	if err := os.WriteFile(path, []byte(`{"listen":":1234"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if want := len(Default().Models); len(cfg.Models) != want {
		t.Fatalf("len(models) = %d, want %d built-ins", len(cfg.Models), want)
	}
}

func TestAuthKeySet(t *testing.T) {
	cfg := Default()
	cfg.APIKeys = []string{"sk-one", "sk-two", "", "  "}
	set := cfg.AuthKeySet()
	want := []string{"sk-one", "sk-two"}
	if len(set) != len(want) {
		t.Fatalf("set = %v, want keys %v", set, want)
	}
	for _, k := range want {
		if _, ok := set[k]; !ok {
			t.Fatalf("missing key %q in %v", k, set)
		}
	}
	if _, ok := set[""]; ok {
		t.Fatal("blank key must not be accepted")
	}
	cfg.APIKeys = nil
	if got := cfg.AuthKeySet(); len(got) != 0 {
		t.Fatalf("expected empty set, got %v", got)
	}
}

func TestMaxConcurrentDefaultsToModelCount(t *testing.T) {
	cfg := Default()
	if cfg.MaxConcurrent != 0 {
		t.Fatalf("zero value should mean auto, got %d", cfg.MaxConcurrent)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatal(err)
	}
	if want := len(cfg.Models); cfg.MaxConcurrent != want {
		t.Fatalf("resolved max_concurrent_requests = %d, want %d (one per model)",
			cfg.MaxConcurrent, want)
	}

	// An explicit value survives validation unchanged.
	explicit := Default()
	explicit.MaxConcurrent = 1
	if err := explicit.Validate(); err != nil {
		t.Fatal(err)
	}
	if explicit.MaxConcurrent != 1 {
		t.Fatalf("explicit max_concurrent_requests = %d, want 1", explicit.MaxConcurrent)
	}
}

func TestValidateRejectsDuplicates(t *testing.T) {
	cfg := Default()
	cfg.Models = []RegistryModel{{ID: "a", File: "a.gguf"}, {ID: "a", File: "b.gguf"}}
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected duplicate id validation error")
	}
}

func TestSlugify(t *testing.T) {
	cases := map[string]string{
		"My Model-v2.Q4_K_M": "my-model-v2.q4-k-m",
		"Qwen2.5 Coder":      "qwen2.5-coder",
	}
	for in, want := range cases {
		if got := slugify(in); got != want {
			t.Errorf("slugify(%q) = %q, want %q", in, got, want)
		}
	}
}
