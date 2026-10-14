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
