package server

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestResponsesNonStream(t *testing.T) {
	ts := newTestServer(t, testConfig())
	resp := postJSON(t, ts.URL+"/v1/responses", `{"model":"alpha","input":"hello responses"}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d body=%s", resp.StatusCode, b)
	}

	var out responsesResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(out.ID, "resp_") {
		t.Fatalf("id = %q, want resp_ prefix", out.ID)
	}
	if out.Object != "response" {
		t.Fatalf("object = %q, want response", out.Object)
	}
	if out.Status != "completed" {
		t.Fatalf("status = %q, want completed", out.Status)
	}
	if out.Model != "alpha" {
		t.Fatalf("model = %q, want alpha", out.Model)
	}
	if out.CreatedAt == 0 {
		t.Fatal("created_at is zero")
	}
	if len(out.Output) != 1 {
		t.Fatalf("output items = %d, want 1", len(out.Output))
	}
	item := out.Output[0]
	if item.Type != "message" || item.Role != "assistant" || item.Status != "completed" {
		t.Fatalf("unexpected output item: %+v", item)
	}
	if len(item.Content) != 1 {
		t.Fatalf("content parts = %d, want 1", len(item.Content))
	}
	part := item.Content[0]
	if part.Type != "output_text" {
		t.Fatalf("content type = %q, want output_text", part.Type)
	}
	if !strings.Contains(part.Text, "stub:alpha") {
		t.Fatalf("output text = %q", part.Text)
	}
	if out.OutputText != part.Text {
		t.Fatalf("output_text = %q, want %q", out.OutputText, part.Text)
	}
	if out.Usage.InputTokens < 1 || out.Usage.OutputTokens < 1 {
		t.Fatalf("unexpected usage: %+v", out.Usage)
	}
	if out.Usage.TotalTokens != out.Usage.InputTokens+out.Usage.OutputTokens {
		t.Fatalf("total_tokens = %d, want %d", out.Usage.TotalTokens,
			out.Usage.InputTokens+out.Usage.OutputTokens)
	}
	if out.Error != nil || out.IncompleteDetails != nil {
		t.Fatalf("successful response carries error/incomplete details: %+v", out)
	}
}

// TestResponsesInputItemsAndInstructions covers the Responses API's array input
// form (typed content parts) plus a top-level instructions string.
func TestResponsesInputItemsAndInstructions(t *testing.T) {
	ts := newTestServer(t, testConfig())
	body := `{
		"model": "alpha",
		"instructions": "You are terse.",
		"input": [
			{"role": "developer", "content": [{"type": "input_text", "text": "Be brief."}]},
			{"role": "user", "content": [{"type": "input_text", "text": "hello items"}]}
		]
	}`
	resp := postJSON(t, ts.URL+"/v1/responses", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d body=%s", resp.StatusCode, b)
	}
	var out responsesResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if len(out.Output) != 1 || len(out.Output[0].Content) != 1 {
		t.Fatalf("unexpected output: %+v", out.Output)
	}
	if !strings.Contains(out.Output[0].Content[0].Text, "stub:alpha") {
		t.Fatalf("output text = %q", out.Output[0].Content[0].Text)
	}
}

func TestResponsesStream(t *testing.T) {
	ts := newTestServer(t, testConfig())
	resp := postJSON(t, ts.URL+"/v1/responses", `{"model":"alpha","input":"stream me","stream":true}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d body=%s", resp.StatusCode, b)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type = %q", ct)
	}

	events := readTestSSE(t, resp.Body)
	byName := map[string][]testSSEEvent{}
	var deltas strings.Builder
	for _, ev := range events {
		if ev.Data == "[DONE]" {
			t.Fatal("Responses streams must not send a [DONE] sentinel")
		}
		byName[ev.Name] = append(byName[ev.Name], ev)
		if ev.Name == "response.output_text.delta" {
			var d struct {
				Type  string `json:"type"`
				Delta string `json:"delta"`
			}
			if err := json.Unmarshal([]byte(ev.Data), &d); err != nil {
				t.Fatalf("bad delta %q: %v", ev.Data, err)
			}
			if d.Type != "response.output_text.delta" {
				t.Fatalf("delta type = %q", d.Type)
			}
			deltas.WriteString(d.Delta)
		}
	}

	for _, name := range []string{
		"response.created",
		"response.in_progress",
		"response.output_item.added",
		"response.content_part.added",
		"response.output_text.delta",
		"response.output_text.done",
		"response.content_part.done",
		"response.output_item.done",
		"response.completed",
	} {
		if len(byName[name]) == 0 {
			t.Fatalf("missing %s event (got %d events)", name, len(events))
		}
	}
	if !strings.Contains(deltas.String(), "stub:alpha") {
		t.Fatalf("streamed text = %q", deltas.String())
	}

	var completed struct {
		Type     string            `json:"type"`
		Response responsesResponse `json:"response"`
	}
	if err := json.Unmarshal([]byte(byName["response.completed"][0].Data), &completed); err != nil {
		t.Fatal(err)
	}
	if completed.Type != "response.completed" || completed.Response.Status != "completed" {
		t.Fatalf("unexpected completed event: %+v", completed)
	}
	if !strings.Contains(completed.Response.OutputText, "stub:alpha") {
		t.Fatalf("completed output_text = %q", completed.Response.OutputText)
	}
	if completed.Response.Usage.TotalTokens < 1 {
		t.Fatalf("unexpected usage: %+v", completed.Response.Usage)
	}
}

func TestResponsesValidation(t *testing.T) {
	ts := newTestServer(t, testConfig())

	t.Run("unknown model", func(t *testing.T) {
		resp := postJSON(t, ts.URL+"/v1/responses", `{"model":"nope","input":"x"}`)
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusNotFound {
			t.Fatalf("status = %d, want 404", resp.StatusCode)
		}
		var eb errorBody
		if err := json.NewDecoder(resp.Body).Decode(&eb); err != nil {
			t.Fatal(err)
		}
		if eb.Error.Code != "model_not_found" {
			t.Fatalf("error code = %q", eb.Error.Code)
		}
	})

	t.Run("missing input", func(t *testing.T) {
		resp := postJSON(t, ts.URL+"/v1/responses", `{"model":"alpha"}`)
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400", resp.StatusCode)
		}
	})

	t.Run("method not allowed", func(t *testing.T) {
		resp, err := http.Get(ts.URL + "/v1/responses")
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusMethodNotAllowed {
			t.Fatalf("status = %d, want 405", resp.StatusCode)
		}
	})
}

func TestResponsesAuthEnforced(t *testing.T) {
	cfg := testConfig()
	cfg.APIKeys = []string{"secret"}
	ts := newTestServer(t, cfg)

	resp := postJSON(t, ts.URL+"/v1/responses", `{"model":"alpha","input":"x"}`)
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", resp.StatusCode)
	}

	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/responses",
		strings.NewReader(`{"model":"alpha","input":"x"}`))
	req.Header.Set("Authorization", "Bearer secret")
	resp2, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("status with key = %d, want 200", resp2.StatusCode)
	}
}

// TestResponsesIncompleteOnMaxOutputTokens verifies that hitting the requested
// output budget marks the response incomplete, as the Responses API specifies.
func TestResponsesIncompleteOnMaxOutputTokens(t *testing.T) {
	ts := newTestServer(t, testConfig())
	resp := postJSON(t, ts.URL+"/v1/responses",
		`{"model":"alpha","input":"tell me a long story","max_output_tokens":4}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d body=%s", resp.StatusCode, b)
	}
	var out responsesResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Status != "incomplete" {
		t.Fatalf("status = %q, want incomplete", out.Status)
	}
	if out.IncompleteDetails == nil || out.IncompleteDetails.Reason != "max_output_tokens" {
		t.Fatalf("incomplete_details = %+v", out.IncompleteDetails)
	}
}
