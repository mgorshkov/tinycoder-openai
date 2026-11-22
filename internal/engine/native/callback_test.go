//go:build !tinycoder_stub

package native

import (
	"testing"
	"time"
)

// TestNoTokenExpired pins the inactivity decision the abort predicate uses: the
// window is measured from the last activity (token or prefill progress), a zero
// timeout disables it, and a genCtx that never recorded activity never expires.
func TestNoTokenExpired(t *testing.T) {
	now := time.Now()
	tests := []struct {
		name    string
		timeout time.Duration
		last    time.Time // zero means "no activity recorded"
		want    bool
	}{
		{name: "inside window", timeout: 60 * time.Second, last: now.Add(-59 * time.Second), want: false},
		{name: "past window", timeout: 60 * time.Second, last: now.Add(-61 * time.Second), want: true},
		{name: "zero timeout disables", timeout: 0, last: now.Add(-time.Hour), want: false},
		{name: "no activity recorded", timeout: 60 * time.Second, last: time.Time{}, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gc := &genCtx{noTokenTimeout: tt.timeout}
			if !tt.last.IsZero() {
				gc.lastActivity.Store(tt.last.UnixNano())
			}
			if got := gc.noTokenExpired(now); got != tt.want {
				t.Fatalf("noTokenExpired() = %v, want %v", got, tt.want)
			}
		})
	}
}
