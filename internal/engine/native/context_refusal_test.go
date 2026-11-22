//go:build !tinycoder_stub

package native

import "testing"

// TestContextRefusal pins the empty-success fix: a zero-token generation is
// reported as a context-length refusal only when the prompt is known to exceed
// a known window. An unknown prompt size or window must not fabricate a
// refusal, and a partial generation is a normal (truncated) result.
func TestContextRefusal(t *testing.T) {
	tests := []struct {
		name         string
		generated    int
		promptTokens int
		window       int
		want         bool
	}{
		{name: "unknown prompt size", generated: 0, promptTokens: 0, window: 1024, want: false},
		{name: "prompt over window", generated: 0, promptTokens: 1025, window: 1024, want: true},
		{name: "prompt exactly at window", generated: 0, promptTokens: 1024, window: 1024, want: false},
		{name: "partial generation is not a refusal", generated: 5, promptTokens: 5000, window: 1024, want: false},
		{name: "unknown window", generated: 0, promptTokens: 1, window: 0, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := contextRefusal(tt.generated, tt.promptTokens, tt.window); got != tt.want {
				t.Fatalf("contextRefusal(%d, %d, %d) = %v, want %v",
					tt.generated, tt.promptTokens, tt.window, got, tt.want)
			}
		})
	}
}
