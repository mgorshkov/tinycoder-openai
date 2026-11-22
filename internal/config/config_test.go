package config

import (
	"os"
	"path/filepath"
	"strings"
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

// TestApplyModelFlagsDerivesID verifies that --model alone yields a one-entry
// registry whose id is slugified from the file base name (without extension)
// and whose name keeps the original base name.
func TestApplyModelFlagsDerivesID(t *testing.T) {
	cfg := Default()
	if err := cfg.ApplyModelFlags("/x/Ternary-Bonsai-2-27B-PQ2_0.gguf", "", 0); err != nil {
		t.Fatal(err)
	}
	if len(cfg.Models) != 1 {
		t.Fatalf("len(models) = %d, want 1: %+v", len(cfg.Models), cfg.Models)
	}
	m := cfg.Models[0]
	if m.ID != "ternary-bonsai-2-27b-pq2-0" {
		t.Fatalf("derived id = %q, want %q", m.ID, "ternary-bonsai-2-27b-pq2-0")
	}
	if m.Name != "Ternary-Bonsai-2-27B-PQ2_0" {
		t.Fatalf("name = %q, want %q", m.Name, "Ternary-Bonsai-2-27B-PQ2_0")
	}
	if m.File != "/x/Ternary-Bonsai-2-27B-PQ2_0.gguf" {
		t.Fatalf("file = %q, want the absolute path unchanged", m.File)
	}
	if m.ContextLength != 0 {
		t.Fatalf("context_length = %d, want 0 (unset -> engine default)", m.ContextLength)
	}
}

// TestApplyModelFlagsExplicitID verifies that --model-id and --ctx are
// forwarded into the single generated registry entry.
func TestApplyModelFlagsExplicitID(t *testing.T) {
	cfg := Default()
	path := "/data/models/gemma/gemma4-coding-Q2_K.gguf"
	if err := cfg.ApplyModelFlags(path, "solo", 4096); err != nil {
		t.Fatal(err)
	}
	if len(cfg.Models) != 1 {
		t.Fatalf("len(models) = %d, want 1: %+v", len(cfg.Models), cfg.Models)
	}
	m := cfg.Models[0]
	if m.ID != "solo" {
		t.Fatalf("id = %q, want %q", m.ID, "solo")
	}
	if m.Name != "gemma4-coding-Q2_K" {
		t.Fatalf("name = %q, want %q", m.Name, "gemma4-coding-Q2_K")
	}
	if m.File != path {
		t.Fatalf("file = %q, want %q", m.File, path)
	}
	if m.ContextLength != 4096 {
		t.Fatalf("context_length = %d, want 4096", m.ContextLength)
	}
}

// TestApplyModelFlagsRelativePathMadeAbsolute verifies that a relative --model
// is resolved against the process working directory (filepath.Abs), never
// against models_dir.
func TestApplyModelFlagsRelativePathMadeAbsolute(t *testing.T) {
	cfg := Default()
	cfg.ModelsDir = "/somewhere/else"
	rel := filepath.Join("scratch", "rel-model.gguf")
	if err := cfg.ApplyModelFlags(rel, "", 0); err != nil {
		t.Fatal(err)
	}
	want, err := filepath.Abs(rel)
	if err != nil {
		t.Fatal(err)
	}
	got := cfg.Models[0].File
	if got != want {
		t.Fatalf("file = %q, want %q (absolute, resolved against the working directory)", got, want)
	}
	if !filepath.IsAbs(got) {
		t.Fatalf("file = %q, want an absolute path", got)
	}
	if strings.HasPrefix(got, cfg.ModelsDir) {
		t.Fatalf("file = %q, must not be resolved against models_dir %q", got, cfg.ModelsDir)
	}
}

// TestApplyModelFlagsErrors covers the flag validation errors: --model-id and
// --ctx are meaningless without --model, and a negative --ctx is invalid.
func TestApplyModelFlagsErrors(t *testing.T) {
	tests := []struct {
		name    string
		model   string
		modelID string
		ctx     int
		wantErr string
	}{
		{name: "model-id without model", modelID: "solo", wantErr: "--model-id/--ctx require --model"},
		{name: "ctx without model", ctx: 4096, wantErr: "--model-id/--ctx require --model"},
		{name: "negative ctx", model: "/x/m.gguf", ctx: -1, wantErr: "--ctx must be >= 0"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Default()
			before := len(cfg.Models)
			err := cfg.ApplyModelFlags(tc.model, tc.modelID, tc.ctx)
			if err == nil {
				t.Fatalf("ApplyModelFlags(%q, %q, %d) = nil, want an error containing %q",
					tc.model, tc.modelID, tc.ctx, tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error = %q, want it to contain %q", err, tc.wantErr)
			}
			if len(cfg.Models) != before {
				t.Fatalf("a failed call changed the registry: len = %d, want %d", len(cfg.Models), before)
			}
		})
	}
}

// TestApplyModelFlagsNoModelIsNoop verifies that no model flag at all leaves the
// configuration untouched.
func TestApplyModelFlagsNoModelIsNoop(t *testing.T) {
	cfg := Default()
	before := append([]RegistryModel(nil), cfg.Models...)
	if err := cfg.ApplyModelFlags("", "", 0); err != nil {
		t.Fatalf("ApplyModelFlags(\"\", \"\", 0) = %v, want nil", err)
	}
	if len(cfg.Models) != len(before) {
		t.Fatalf("len(models) = %d, want %d", len(cfg.Models), len(before))
	}
	for i := range before {
		if cfg.Models[i] != before[i] {
			t.Fatalf("models[%d] = %+v, want the unchanged built-in %+v", i, cfg.Models[i], before[i])
		}
	}
}

// TestApplyModelFlagsReplacesRegistry verifies that --model replaces the whole
// registry with exactly one entry and that fields the flags omit stay zero,
// even when the config file's model carries architecture/quantization/etc.
func TestApplyModelFlagsReplacesRegistry(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.json")
	body := `{"listen":"127.0.0.1:9999","models":[` +
		`{"id":"from-file","name":"From File","file":"from-file.gguf","architecture":"qwen2",` +
		`"quantization":"Q4_K_M","context_length":8192,"n_threads":7,"size_bytes":123,` +
		`"description":"file description"}]}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := cfg.ApplyModelFlags("/models/solo.gguf", "solo", 4096); err != nil {
		t.Fatal(err)
	}
	if len(cfg.Models) != 1 {
		t.Fatalf("len(models) = %d, want exactly 1: %+v", len(cfg.Models), cfg.Models)
	}
	m := cfg.Models[0]
	if m.ID != "solo" || m.Name != "solo" || m.File != "/models/solo.gguf" || m.ContextLength != 4096 {
		t.Fatalf("entry = %+v, want {ID:solo Name:solo File:/models/solo.gguf ContextLength:4096}", m)
	}
	if m.Architecture != "" || m.Quantization != "" || m.NThreads != 0 ||
		m.SizeBytes != 0 || m.Description != "" {
		t.Fatalf("file-model fields leaked into the flag registry: %+v", m)
	}
	// Non-registry configuration is untouched by model flags.
	if cfg.Listen != "127.0.0.1:9999" {
		t.Fatalf("listen = %q, want 127.0.0.1:9999", cfg.Listen)
	}
}

// TestPreloadIDs covers target selection for the startup warm-up: an explicit
// preload_models list wins, preload:true selects the single registered model,
// and neither selector configured yields no targets (lazy behavior).
func TestPreloadIDs(t *testing.T) {
	cfg := Default()
	cfg.Models = []RegistryModel{{ID: "solo", Name: "Solo", File: "solo.gguf"}}
	if got := cfg.PreloadIDs(); got != nil {
		t.Fatalf("PreloadIDs() with preload disabled = %v, want nil", got)
	}

	cfg.Preload = true
	if got := cfg.PreloadIDs(); len(got) != 1 || got[0] != "solo" {
		t.Fatalf("PreloadIDs() with preload:true and one model = %v, want [solo]", got)
	}

	// An explicit list wins over preload:true and may name several models.
	cfg.Models = []RegistryModel{{ID: "a"}, {ID: "b"}}
	cfg.PreloadModels = []string{"b", "a"}
	if got := cfg.PreloadIDs(); len(got) != 2 || got[0] != "b" || got[1] != "a" {
		t.Fatalf("PreloadIDs() with an explicit list = %v, want [b a]", got)
	}

	// An empty list falls back to the preload selectors; with preload off there
	// are no targets.
	cfg.PreloadModels = []string{}
	cfg.Preload = false
	if got := cfg.PreloadIDs(); got != nil {
		t.Fatalf("PreloadIDs() with an empty list and preload disabled = %v, want nil", got)
	}
}

// TestValidatePreload covers the preload consistency checks: every named model
// must exist, preload:true needs an unambiguous target, and a negative timeout
// is rejected.
func TestValidatePreload(t *testing.T) {
	t.Run("unknown preload_models id", func(t *testing.T) {
		cfg := Default()
		cfg.Models = []RegistryModel{{ID: "alpha", File: "alpha.gguf"}}
		cfg.PreloadModels = []string{"nope"}
		err := cfg.Validate()
		if err == nil || !strings.Contains(err.Error(), `preload_models: unknown model id "nope"`) {
			t.Fatalf("Validate() = %v, want an unknown model id error", err)
		}
	})

	t.Run("preload several models without a list", func(t *testing.T) {
		cfg := Default()
		cfg.Models = []RegistryModel{{ID: "alpha", File: "alpha.gguf"}, {ID: "beta", File: "beta.gguf"}}
		cfg.Preload = true
		err := cfg.Validate()
		if err == nil || !strings.Contains(err.Error(), "preload requires a single registered model or preload_models") {
			t.Fatalf("Validate() = %v, want a preload target error", err)
		}
	})

	t.Run("negative timeout", func(t *testing.T) {
		cfg := Default()
		cfg.Models = []RegistryModel{{ID: "alpha", File: "alpha.gguf"}}
		cfg.PreloadTimeoutSec = -1
		if err := cfg.Validate(); err == nil {
			t.Fatal("Validate() = nil, want a negative preload_timeout_sec error")
		}
	})

	t.Run("valid explicit list for several models", func(t *testing.T) {
		cfg := Default()
		cfg.Models = []RegistryModel{{ID: "alpha", File: "alpha.gguf"}, {ID: "beta", File: "beta.gguf"}}
		cfg.PreloadModels = []string{"alpha"}
		if err := cfg.Validate(); err != nil {
			t.Fatalf("Validate() = %v, want nil", err)
		}
	})
}

// TestPreloadTimeoutDefaultAndEnv verifies the warm-up budget default and its
// environment overrides, including the preload toggle and target list.
func TestPreloadTimeoutDefaultAndEnv(t *testing.T) {
	cfg := Default()
	if cfg.PreloadTimeoutSec != 300 {
		t.Fatalf("default preload_timeout_sec = %d, want 300", cfg.PreloadTimeoutSec)
	}

	t.Setenv("TINYCODER_PRELOAD_TIMEOUT_SEC", "42")
	t.Setenv("TINYCODER_PRELOAD", "true")
	t.Setenv("TINYCODER_PRELOAD_MODELS", "alpha, beta")
	if err := cfg.ApplyEnv(); err != nil {
		t.Fatal(err)
	}
	if cfg.PreloadTimeoutSec != 42 {
		t.Fatalf("env preload_timeout_sec = %d, want 42", cfg.PreloadTimeoutSec)
	}
	if !cfg.Preload {
		t.Fatal("env TINYCODER_PRELOAD=true did not set Preload")
	}
	if len(cfg.PreloadModels) != 2 || cfg.PreloadModels[0] != "alpha" || cfg.PreloadModels[1] != "beta" {
		t.Fatalf("env TINYCODER_PRELOAD_MODELS = %v, want [alpha beta]", cfg.PreloadModels)
	}
}

// TestExistingConfigWithoutPreloadKeysValidates verifies that a config written
// before the preload keys existed keeps the defaults and still validates.
func TestExistingConfigWithoutPreloadKeysValidates(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.json")
	body := `{"listen":":1234","models":[{"id":"solo","name":"Solo","file":"solo.gguf"}]}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Preload || cfg.PreloadModels != nil || cfg.PreloadTimeoutSec != 300 {
		t.Fatalf("preload defaults not applied: preload=%v preload_models=%v timeout=%d",
			cfg.Preload, cfg.PreloadModels, cfg.PreloadTimeoutSec)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() = %v, want nil for a pre-preload config", err)
	}
}
