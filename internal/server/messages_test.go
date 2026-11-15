package server

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestMessagesNonStream(t *testing.T) {
	ts := newTestServer(t, testConfig())
	body := `{"model":"alpha","max_tokens":64,"messages":[{"role":"user","content":"hello there"}]}`
	resp := postJSON(t, ts.URL+"/v1/messages", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d body=%s", resp.StatusCode, b)
	}

	var out messagesResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(out.ID, "msg_") {
		t.Fatalf("id = %q, want msg_ prefix", out.ID)
	}
	if out.Type != "message" || out.Role != "assistant" || out.Model != "alpha" {
		t.Fatalf("unexpected response: %+v", out)
	}
	if len(out.Content) != 1 || out.Content[0].Type != "text" {
		t.Fatalf("unexpected content: %+v", out.Content)
	}
	if !strings.Contains(out.Content[0].Text, "stub:alpha") {
		t.Fatalf("text = %q", out.Content[0].Text)
	}
	if out.StopReason != "end_turn" {
		t.Fatalf("stop_reason = %q, want end_turn", out.StopReason)
	}
	if out.Usage.InputTokens < 1 || out.Usage.OutputTokens < 1 {
		t.Fatalf("unexpected usage: %+v", out.Usage)
	}
}

// TestMessagesSystemAndBlockContent covers Anthropic's top-level system prompt
// and array-form message content.
func TestMessagesSystemAndBlockContent(t *testing.T) {
	ts := newTestServer(t, testConfig())
	body := `{
		"model": "alpha",
		"max_tokens": 32,
		"system": [{"type": "text", "text": "Be nice."}],
		"messages": [{"role": "user", "content": [{"type": "text", "text": "hi blocks"}]}]
	}`
	resp := postJSON(t, ts.URL+"/v1/messages", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d body=%s", resp.StatusCode, b)
	}
	var out messagesResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if len(out.Content) != 1 || !strings.Contains(out.Content[0].Text, "stub:alpha") {
		t.Fatalf("unexpected content: %+v", out.Content)
	}
}

func TestMessagesMissingMaxTokensRejected(t *testing.T) {
	ts := newTestServer(t, testConfig())
	resp := postJSON(t, ts.URL+"/v1/messages",
		`{"model":"alpha","messages":[{"role":"user","content":"hi"}]}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
	var eb anthropicErrorBody
	if err := json.NewDecoder(resp.Body).Decode(&eb); err != nil {
		t.Fatal(err)
	}
	if eb.Type != "error" || eb.Error.Type != "invalid_request_error" {
		t.Fatalf("unexpected error body: %+v", eb)
	}
	if !strings.Contains(eb.Error.Message, "max_tokens") {
		t.Fatalf("message = %q", eb.Error.Message)
	}
}

func TestMessagesUnknownModel404(t *testing.T) {
	ts := newTestServer(t, testConfig())
	resp := postJSON(t, ts.URL+"/v1/messages",
		`{"model":"nope","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", resp.StatusCode)
	}
	var eb anthropicErrorBody
	if err := json.NewDecoder(resp.Body).Decode(&eb); err != nil {
		t.Fatal(err)
	}
	if eb.Type != "error" || eb.Error.Type != "not_found_error" {
		t.Fatalf("unexpected error body: %+v", eb)
	}
}

func TestMessagesStream(t *testing.T) {
	ts := newTestServer(t, testConfig())
	body := `{"model":"alpha","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"stream me"}]}`
	resp := postJSON(t, ts.URL+"/v1/messages", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d body=%s", resp.StatusCode, b)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type = %q", ct)
	}

	events := readTestSSE(t, resp.Body)
	byName := map[string][]map[string]any{}
	var text strings.Builder
	for _, ev := range events {
		if ev.Data == "[DONE]" {
			t.Fatal("Messages streams must not send a [DONE] sentinel")
		}
		var payload map[string]any
		if err := json.Unmarshal([]byte(ev.Data), &payload); err != nil {
			t.Fatalf("bad event %q: %v", ev.Data, err)
		}
		if got, _ := payload["type"].(string); ev.Name != "" && got != ev.Name {
			t.Fatalf("event name %q != data type %q", ev.Name, got)
		}
		byName[ev.Name] = append(byName[ev.Name], payload)
		if ev.Name == "content_block_delta" {
			delta := payload["delta"].(map[string]any)
			if delta["type"] != "text_delta" {
				t.Fatalf("delta type = %v", delta["type"])
			}
			text.WriteString(delta["text"].(string))
		}
	}

	for _, name := range []string{
		"message_start", "content_block_start", "content_block_delta",
		"content_block_stop", "message_delta", "message_stop",
	} {
		if len(byName[name]) == 0 {
			t.Fatalf("missing %s event (got %d events)", name, len(events))
		}
	}
	if !strings.Contains(text.String(), "stub:alpha") {
		t.Fatalf("streamed text = %q", text.String())
	}

	md := byName["message_delta"][0]
	delta := md["delta"].(map[string]any)
	if delta["stop_reason"] != "end_turn" {
		t.Fatalf("stop_reason = %v", delta["stop_reason"])
	}
	usage := md["usage"].(map[string]any)
	if out, _ := usage["output_tokens"].(float64); out < 1 {
		t.Fatalf("usage = %+v", usage)
	}
}

func TestMessagesAuthUsesAnthropicErrorShape(t *testing.T) {
	cfg := testConfig()
	cfg.APIKeys = []string{"secret"}
	ts := newTestServer(t, cfg)

	resp := postJSON(t, ts.URL+"/v1/messages",
		`{"model":"alpha","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", resp.StatusCode)
	}
	var eb anthropicErrorBody
	if err := json.NewDecoder(resp.Body).Decode(&eb); err != nil {
		t.Fatal(err)
	}
	if eb.Type != "error" || eb.Error.Type != "authentication_error" {
		t.Fatalf("unexpected error body: %+v", eb)
	}
}

func TestMessagesMethodNotAllowed(t *testing.T) {
	ts := newTestServer(t, testConfig())
	resp, err := http.Get(ts.URL + "/v1/messages")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", resp.StatusCode)
	}
}

// TestMessagesStopReasonMaxTokens verifies that hitting the requested token
// budget is reported as stop_reason "max_tokens" so clients know to continue.
func TestMessagesStopReasonMaxTokens(t *testing.T) {
	ts := newTestServer(t, testConfig())
	body := `{"model":"alpha","max_tokens":4,"messages":[{"role":"user","content":"tell me a long story"}]}`
	resp := postJSON(t, ts.URL+"/v1/messages", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d body=%s", resp.StatusCode, b)
	}
	var out messagesResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.StopReason != "max_tokens" {
		t.Fatalf("stop_reason = %q, want max_tokens", out.StopReason)
	}
}
