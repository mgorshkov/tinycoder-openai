package server

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

// TestSpecDocumentsAllEndpoints guards the served OpenAPI document against
// drifting from the routed API surface.
func TestSpecDocumentsAllEndpoints(t *testing.T) {
	ts := newTestServer(t, testConfig())

	resp, err := http.Get(ts.URL + "/openai.json")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	var doc struct {
		Paths      map[string]json.RawMessage `json:"paths"`
		Components struct {
			Schemas map[string]json.RawMessage `json:"schemas"`
		} `json:"components"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&doc); err != nil {
		t.Fatalf("spec is not valid JSON: %v", err)
	}
	for _, path := range []string{"/models", "/chat/completions", "/completions", "/responses", "/messages"} {
		if _, ok := doc.Paths[path]; !ok {
			t.Fatalf("spec does not document path %s", path)
		}
	}
	for _, name := range []string{"ResponsesRequest", "ResponsesResponse", "MessagesRequest", "MessagesResponse"} {
		if _, ok := doc.Components.Schemas[name]; !ok {
			t.Fatalf("spec does not define schema %s", name)
		}
	}

	yresp, err := http.Get(ts.URL + "/openai.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer yresp.Body.Close()
	yaml, _ := io.ReadAll(yresp.Body)
	for _, want := range []string{"/responses:", "/messages:"} {
		if !strings.Contains(string(yaml), want) {
			t.Fatalf("YAML spec does not document %s", want)
		}
	}
}
