package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/mgorshkov/tinycoder-openai/internal/engine"
)

// This file implements the OpenAI Responses API (POST /v1/responses).
//
// The endpoint is a stateless translation layer: `input` (a string or an array
// of input items) and `instructions` are rendered into the engine's chat
// messages, and the generated text is returned as a `message` output item with
// an `output_text` content part. Fields the gateway does not model (store,
// previous_response_id, reasoning, text.format, tools, ...) are accepted and
// ignored so official Responses clients keep working.

// responsesRequest is the POST /v1/responses body.
type responsesRequest struct {
	Model           string         `json:"model"`
	Input           responsesInput `json:"input"`
	Instructions    string         `json:"instructions,omitempty"`
	MaxOutputTokens *int           `json:"max_output_tokens,omitempty"`
	Temperature     *float32       `json:"temperature,omitempty"`
	TopP            *float32       `json:"top_p,omitempty"`
	TopK            *float32       `json:"top_k,omitempty"`
	Stream          bool           `json:"stream,omitempty"`
}

// responsesInput accepts either a bare string or an array of input items.
type responsesInput struct {
	text  string
	items []responsesInputItem
}

func (in *responsesInput) UnmarshalJSON(b []byte) error {
	b = bytes.TrimSpace(b)
	if len(b) == 0 || string(b) == "null" {
		return nil
	}
	if b[0] == '"' {
		return json.Unmarshal(b, &in.text)
	}
	if b[0] != '[' {
		return errors.New("input must be a string or an array of input items")
	}
	return json.Unmarshal(b, &in.items)
}

// responsesInputItem is one entry of an array-form input. Only `message` items
// (the default when type is empty) carry promotable text.
type responsesInputItem struct {
	Type    string          `json:"type,omitempty"`
	Role    string          `json:"role,omitempty"`
	Content json.RawMessage `json:"content,omitempty"`
}

// messages renders the request's input and instructions into engine messages.
func (r responsesRequest) messages() ([]engine.Message, error) {
	var out []engine.Message
	if strings.TrimSpace(r.Input.text) != "" {
		out = append(out, engine.Message{Role: "user", Content: r.Input.text})
	}
	for _, it := range r.Input.items {
		if it.Type != "" && it.Type != "message" {
			// function_call / function_call_output / reasoning items carry no
			// promotable text for a text-only gateway.
			continue
		}
		var c messageContent
		if len(it.Content) > 0 {
			if err := c.UnmarshalJSON(it.Content); err != nil {
				return nil, errors.New("invalid input item content: " + err.Error())
			}
		}
		if strings.TrimSpace(c.text) == "" {
			continue
		}
		role := it.Role
		if role == "" {
			role = "user"
		}
		if role == "developer" {
			role = "system"
		}
		out = append(out, engine.Message{Role: role, Content: c.text})
	}
	if len(out) == 0 {
		return nil, errors.New("field 'input' is required")
	}
	injectSystem(&out, strings.TrimSpace(r.Instructions))
	return out, nil
}

// handleResponses serves POST /v1/responses.
func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error", "")
		return
	}
	if !s.authorize(w, r) {
		return
	}

	var req responsesRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8<<20))
	if err := dec.Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error(), "invalid_request_error", "")
		return
	}

	modelID, genReq, err := s.responsesEngineRequest(req)
	if err != nil {
		writeResponsesRequestError(w, err)
		return
	}

	// One request context bounds the whole request: the prompt-window
	// preflight, any model load, and generation (including streaming).
	ctx, cancel := s.requestContext(r)
	defer cancel()

	// Reject an oversized prompt before opening a stream or generating. The
	// streaming check only runs for a warm, idle model (R1).
	if err := s.checkContextWindow(ctx, modelID, genReq, req.Stream); err != nil {
		writeResponsesRequestError(w, err)
		return
	}

	if req.Stream {
		s.streamResponses(w, ctx, modelID, genReq)
		return
	}

	text, usage, genErr := s.generateText(ctx, modelID, genReq)
	if genErr != nil {
		s.writeGenerateError(w, genErr)
		return
	}
	writeJSON(w, http.StatusOK, newResponsesResponse(modelID, text, usage, genReq))
}

// responsesEngineRequest validates a Responses request and resolves it to an
// engine.GenerateRequest.
func (s *Server) responsesEngineRequest(req responsesRequest) (string, engine.GenerateRequest, error) {
	modelID, err := s.resolveModelID(req.Model)
	if err != nil {
		return modelID, engine.GenerateRequest{}, err
	}
	genReq, err := s.newGenerateRequest(req.MaxOutputTokens, req.Temperature, req.TopP, req.TopK)
	if err != nil {
		return modelID, genReq, err
	}
	msgs, err := req.messages()
	if err != nil {
		return modelID, genReq, err
	}
	genReq.Messages = msgs
	return modelID, genReq, nil
}

// generateText runs one blocking generation on the request context and returns
// its full text.
func (s *Server) generateText(ctx context.Context, modelID string, genReq engine.GenerateRequest) (string, engine.Usage, error) {
	var b strings.Builder
	usage, err := s.mgr.Generate(ctx, modelID, genReq, func(t engine.Token) bool {
		b.WriteString(t.Text)
		return true
	})
	return b.String(), usage, err
}

// writeResponsesRequestError maps request-resolution failures to Responses-style
// error responses.
func writeResponsesRequestError(w http.ResponseWriter, err error) {
	if errors.Is(err, ErrModelNotFound) {
		writeError(w, http.StatusNotFound, err.Error(), "invalid_request_error", "model_not_found")
		return
	}
	if errors.Is(err, engine.ErrContextLength) {
		writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", "context_length_exceeded")
		return
	}
	writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", "")
}

// ---- Response shapes ----

type responsesResponse struct {
	ID                string                `json:"id"`
	Object            string                `json:"object"`
	CreatedAt         int64                 `json:"created_at"`
	Status            string                `json:"status"`
	Error             *responsesError       `json:"error"`
	IncompleteDetails *responsesIncomplete  `json:"incomplete_details"`
	Model             string                `json:"model"`
	Output            []responsesOutputItem `json:"output"`
	OutputText        string                `json:"output_text,omitempty"`
	ParallelToolCalls bool                  `json:"parallel_tool_calls"`
	Temperature       float32               `json:"temperature"`
	TopP              float32               `json:"top_p"`
	ToolChoice        string                `json:"tool_choice"`
	Tools             []any                 `json:"tools"`
	Usage             responsesUsage        `json:"usage"`
}

type responsesError struct {
	Code    string `json:"code,omitempty"`
	Message string `json:"message"`
}

type responsesIncomplete struct {
	Reason string `json:"reason"`
}

type responsesOutputItem struct {
	Type    string                 `json:"type"`
	ID      string                 `json:"id"`
	Status  string                 `json:"status"`
	Role    string                 `json:"role,omitempty"`
	Content []responsesContentPart `json:"content"`
}

type responsesContentPart struct {
	Type        string `json:"type"`
	Text        string `json:"text"`
	Annotations []any  `json:"annotations"`
}

type responsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// newResponsesResponse builds the standard response object. The engine does not
// report a finish reason, so truncation at the token budget is inferred the same
// way OpenAI-compatible servers do: output length reaching max_tokens means the
// response is incomplete.
func newResponsesResponse(modelID, text string, usage engine.Usage, req engine.GenerateRequest) responsesResponse {
	status := "completed"
	var incomplete *responsesIncomplete
	if req.MaxTokens > 0 && usage.CompletionTokens >= req.MaxTokens {
		status = "incomplete"
		incomplete = &responsesIncomplete{Reason: "max_output_tokens"}
	}
	part := responsesContentPart{Type: "output_text", Text: text, Annotations: []any{}}
	return responsesResponse{
		ID:                newIDSep("resp", "_"),
		Object:            "response",
		CreatedAt:         nowUnix(),
		Status:            status,
		IncompleteDetails: incomplete,
		Model:             modelID,
		Output: []responsesOutputItem{{
			Type:    "message",
			ID:      newIDSep("msg", "_"),
			Status:  status,
			Role:    "assistant",
			Content: []responsesContentPart{part},
		}},
		OutputText:  text,
		Tools:       []any{},
		ToolChoice:  "auto",
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Usage: responsesUsage{
			InputTokens:  usage.PromptTokens,
			OutputTokens: usage.CompletionTokens,
			TotalTokens:  usage.TotalTokens,
		},
	}
}

// ---- Streaming ----

// streamResponses emits the Responses API SSE event sequence. Unlike Chat
// Completions there is no [DONE] sentinel: the stream ends after the terminal
// response.completed (or response.failed) event.
func (s *Server) streamResponses(w http.ResponseWriter, ctx context.Context, modelID string, req engine.GenerateRequest) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming unsupported", "server_error", "")
		return
	}
	setSSEHeaders(w)
	sw := newSSEWriter(w, flusher)
	stopHB := sw.startHeartbeat(sseHeartbeat)
	defer stopHB()

	respID := newIDSep("resp", "_")
	msgID := newIDSep("msg", "_")
	created := nowUnix()

	inProgress := responsesResponse{
		ID: respID, Object: "response", CreatedAt: created, Status: "in_progress",
		Model: modelID, Output: []responsesOutputItem{}, Tools: []any{}, ToolChoice: "auto",
		Temperature: req.Temperature, TopP: req.TopP,
	}
	sw.writeEvent("response.created", map[string]any{
		"type": "response.created", "response": inProgress,
	})
	sw.writeEvent("response.in_progress", map[string]any{
		"type": "response.in_progress", "response": inProgress,
	})

	item := responsesOutputItem{
		Type: "message", ID: msgID, Status: "in_progress", Role: "assistant",
		Content: []responsesContentPart{},
	}
	sw.writeEvent("response.output_item.added", map[string]any{
		"type": "response.output_item.added", "output_index": 0, "item": item,
	})
	sw.writeEvent("response.content_part.added", map[string]any{
		"type": "response.content_part.added", "item_id": msgID, "output_index": 0,
		"content_index": 0,
		"part":          responsesContentPart{Type: "output_text", Text: "", Annotations: []any{}},
	})

	var b strings.Builder
	usage, genErr := s.runStream(ctx, modelID, req, func(text string) {
		b.WriteString(text)
		sw.writeEvent("response.output_text.delta", map[string]any{
			"type": "response.output_text.delta", "item_id": msgID, "output_index": 0,
			"content_index": 0, "delta": text,
		})
	})
	if genErr != nil {
		stopHB()
		code := "server_error"
		if errors.Is(genErr, engine.ErrContextLength) {
			code = "context_length_exceeded"
		}
		failed := responsesResponse{
			ID: respID, Object: "response", CreatedAt: created, Status: "failed",
			Model: modelID, Output: []responsesOutputItem{}, Tools: []any{}, ToolChoice: "auto",
			Error: &responsesError{Code: code, Message: "generation failed: " + genErr.Error()},
		}
		sw.writeEvent("response.failed", map[string]any{
			"type": "response.failed", "response": failed,
		})
		return
	}
	stopHB()

	text := b.String()
	part := responsesContentPart{Type: "output_text", Text: text, Annotations: []any{}}
	sw.writeEvent("response.output_text.done", map[string]any{
		"type": "response.output_text.done", "item_id": msgID, "output_index": 0,
		"content_index": 0, "text": text,
	})
	sw.writeEvent("response.content_part.done", map[string]any{
		"type": "response.content_part.done", "item_id": msgID, "output_index": 0,
		"content_index": 0, "part": part,
	})

	final := newResponsesResponse(modelID, text, usage, req)
	final.ID = respID
	final.CreatedAt = created
	final.Output[0].ID = msgID
	sw.writeEvent("response.output_item.done", map[string]any{
		"type": "response.output_item.done", "output_index": 0, "item": final.Output[0],
	})
	sw.writeEvent("response.completed", map[string]any{
		"type": "response.completed", "response": final,
	})
}
