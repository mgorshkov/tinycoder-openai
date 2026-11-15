package server

import (
	"bufio"
	"io"
	"net/http"
	"strings"
	"testing"
)

// testSSEEvent is one parsed Server-Sent Event: its optional event name and the
// JSON payload of its data line.
type testSSEEvent struct {
	Name string
	Data string
}

// readTestSSE parses an SSE response body into events. Comment lines
// (keep-alives) and blank separators are skipped.
func readTestSSE(t *testing.T, r io.Reader) []testSSEEvent {
	t.Helper()
	var (
		out  []testSSEEvent
		name string
	)
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := sc.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			name = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			out = append(out, testSSEEvent{Name: name, Data: strings.TrimPrefix(line, "data: ")})
		case line == "":
			name = ""
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("reading SSE stream: %v", err)
	}
	return out
}

// postJSON issues a POST with a JSON body and returns the response.
func postJSON(t *testing.T, url, body string) *http.Response {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	return resp
}
