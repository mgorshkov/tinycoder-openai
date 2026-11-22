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

// This file implements the Anthropic Messages API (POST /v1/messages).
//
// Like the other generation endpoints it is a stateless translation layer:
// `system` and the `messages` array are rendered into the engine's chat
// messages, and the generated text is returned as a `text` content block.
// Anthropic-specific fields the gateway does not model (tools, tool_choice,
// stop_sequences, metadata, thinking, ...) are accepted and ignored.

// messagesRequest is the POST /v1/messages body. max_tokens is required by the
// Anthropic API, so it is a pointer to distinguish "missing" from zero.
type messagesRequest struct {
	Model       string            `json:"model"`
	MaxTokens   *int              `json:"max_tokens"`
	System      messagesSystem    `json:"system,omitempty"`
	Messages    []messagesMessage `json:"messages"`
	Temperature *float32          `json:"temperature,omitempty"`
	TopP        *float32          `json:"top_p,omitempty"`
	TopK        *float32          `json:"top_k,omitempty"`
	Stream      bool              `json:"stream,omitempty"`
}

// messagesSystem accepts Anthropic's system prompt as either a string or an
// array of text blocks.
type messagesSystem struct {
	text string
}

func (m *messagesSystem) UnmarshalJSON(b []byte) error {
	b = bytes.TrimSpace(b)
	if len(b) == 0 || string(b) == "null" {
		return nil
	}
	if b[0] == '"' {
		return json.Unmarshal(b, &m.text)
	}
	if b[0] != '[' {
		return errors.New("system must be a string or an array of text blocks")
	}
	var parts []contentPart
	if err := json.Unmarshal(b, &parts); err != nil {
		return err
	}
	var sb strings.Builder
	for _, p := range parts {
		if p.Text == "" {
			continue
		}
		if sb.Len() > 0 {
			sb.WriteByte('\n')
		}
		sb.WriteString(p.Text)
	}
	m.text = sb.String()
	return nil
}

// messagesMessage is one conversation turn; content is a string or an array of
// typed blocks (only text blocks carry promotable data).
type messagesMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// engineMessages renders the system prompt and conversation into engine
// messages.
func (r messagesRequest) engineMessages() ([]engine.Message, error) {
	if len(r.Messages) == 0 {
		return nil, errors.New("field 'messages' is required")
	}
	var out []engine.Message
	for _, m := range r.Messages {
		var c messageContent
		if len(m.Content) > 0 {
			if err := c.UnmarshalJSON(m.Content); err != nil {
				return nil, errors.New("invalid message content: " + err.Error())
			}
		}
		if strings.TrimSpace(c.text) == "" {
			continue
		}
		out = append(out, engine.Message{Role: m.Role, Content: c.text})
	}
	if len(out) == 0 {
		return nil, errors.New("field 'messages' is required")
	}
	injectSystem(&out, strings.TrimSpace(r.System.text))
	return out, nil
}

// handleMessages serves POST /v1/messages.
func (s *Server) handleMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeAnthropicError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method not allowed")
		return
	}
	if !s.authorized(r) {
		w.Header().Set("WWW-Authenticate", "Bearer")
		writeAnthropicError(w, http.StatusUnauthorized, "authentication_error", "invalid or missing API key")
		return
	}

	var req messagesRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8<<20))
	if err := dec.Decode(&req); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON body: "+err.Error())
		return
	}

	modelID, genReq, err := s.messagesEngineRequest(req)
	if err != nil {
		status := http.StatusBadRequest
		typ := "invalid_request_error"
		if errors.Is(err, ErrModelNotFound) {
			status = http.StatusNotFound
			typ = "not_found_error"
		}
		writeAnthropicError(w, status, typ, err.Error())
		return
	}

	// One request context bounds request handling, the prompt-window preflight
	// and generation (including streaming). Cancellation is cooperative: a
	// native model load is a blocking C call that does not honor it.
	ctx, cancel := s.requestContext(r)
	defer cancel()

	// Reject an oversized prompt before opening a stream or generating. The
	// streaming check only runs for a warm, idle model (R1); the Anthropic error
	// shape carries a type and message, but no code.
	if err := s.checkContextWindow(ctx, modelID, genReq, req.Stream); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	if req.Stream {
		s.streamMessages(w, ctx, modelID, genReq)
		return
	}

	text, usage, genErr := s.generateText(ctx, modelID, genReq)
	if genErr != nil {
		s.writeMessagesError(w, genErr)
		return
	}
	writeJSON(w, http.StatusOK, newMessagesResponse(modelID, text, usage, genReq.MaxTokens))
}

// messagesEngineRequest validates a Messages request and resolves it to an
// engine.GenerateRequest.
func (s *Server) messagesEngineRequest(req messagesRequest) (string, engine.GenerateRequest, error) {
	modelID, err := s.resolveModelID(req.Model)
	if err != nil {
		return modelID, engine.GenerateRequest{}, err
	}
	if req.MaxTokens == nil {
		return modelID, engine.GenerateRequest{}, errors.New("field 'max_tokens' is required")
	}
	// Anthropic constrains temperature to [0, 1] (unlike OpenAI's [0, 2]).
	if req.Temperature != nil && (*req.Temperature < 0 || *req.Temperature > 1) {
		return modelID, engine.GenerateRequest{}, errors.New("temperature must be between 0 and 1")
	}
	genReq, err := s.newGenerateRequest(req.MaxTokens, req.Temperature, req.TopP, req.TopK)
	if err != nil {
		return modelID, genReq, err
	}
	msgs, err := req.engineMessages()
	if err != nil {
		return modelID, genReq, err
	}
	genReq.Messages = msgs
	return modelID, genReq, nil
}

// writeMessagesError maps a generation failure to an Anthropic-shaped error.
func (s *Server) writeMessagesError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, ErrModelNotFound):
		writeAnthropicError(w, http.StatusNotFound, "not_found_error", err.Error())
	case errors.Is(err, engine.ErrContextLength):
		// The prompt does not fit the model's window (the preflight normally
		// catches this; this also maps an engine-side refusal).
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
	case errors.Is(err, ErrCapacity):
		writeAnthropicError(w, http.StatusServiceUnavailable, "overloaded_error", err.Error())
	case errors.Is(err, engine.ErrNoToken):
		writeAnthropicError(w, http.StatusGatewayTimeout, "api_error", err.Error())
	case errors.Is(err, engine.ErrContextCanceled):
		writeAnthropicError(w, http.StatusGatewayTimeout, "api_error", "generation canceled")
	case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
		// A manager/acquisition deadline (or a shutdown-canceled request) is a
		// 504 on the OpenAI surfaces; keep the Messages API consistent.
		writeAnthropicError(w, http.StatusGatewayTimeout, "api_error", "generation canceled")
	case errors.Is(err, engine.ErrNotLoaded):
		writeAnthropicError(w, http.StatusConflict, "api_error", "model not loaded")
	default:
		s.log.Error("generation failed", "err", err)
		writeAnthropicError(w, http.StatusInternalServerError, "api_error", "generation failed: "+err.Error())
	}
}

// ---- Response shapes ----

type messagesResponse struct {
	ID           string                 `json:"id"`
	Type         string                 `json:"type"`
	Role         string                 `json:"role"`
	Model        string                 `json:"model"`
	Content      []messagesContentBlock `json:"content"`
	StopReason   string                 `json:"stop_reason,omitempty"`
	StopSequence *string                `json:"stop_sequence"`
	Usage        messagesUsage          `json:"usage"`
}

type messagesContentBlock struct {
	Type string `json:"type"`
	// Text is always serialized: Anthropic text blocks carry an explicit text
	// field (empty in content_block_start).
	Text string `json:"text"`
}

type messagesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// newMessagesResponse builds the standard Messages response object. The engine
// does not report a finish reason, so reaching the token budget is reported as
// stop_reason "max_tokens" (clients such as Claude Code use it to continue);
// otherwise the turn ended normally ("end_turn").
func newMessagesResponse(modelID, text string, usage engine.Usage, maxTokens int) messagesResponse {
	stop := "end_turn"
	if maxTokens > 0 && usage.CompletionTokens >= maxTokens {
		stop = "max_tokens"
	}
	return messagesResponse{
		ID:         newIDSep("msg", "_"),
		Type:       "message",
		Role:       "assistant",
		Model:      modelID,
		Content:    []messagesContentBlock{{Type: "text", Text: text}},
		StopReason: stop,
		Usage: messagesUsage{
			InputTokens:  usage.PromptTokens,
			OutputTokens: usage.CompletionTokens,
		},
	}
}

// ---- Errors ----

type anthropicErrorBody struct {
	Type  string               `json:"type"`
	Error anthropicErrorDetail `json:"error"`
}

type anthropicErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

func writeAnthropicError(w http.ResponseWriter, status int, typ, msg string) {
	writeJSON(w, status, anthropicErrorBody{
		Type:  "error",
		Error: anthropicErrorDetail{Type: typ, Message: msg},
	})
}

// ---- Streaming ----

// streamMessages emits the Anthropic Messages SSE event sequence. There is no
// [DONE] sentinel: the stream ends after message_stop.
func (s *Server) streamMessages(w http.ResponseWriter, ctx context.Context, modelID string, req engine.GenerateRequest) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeAnthropicError(w, http.StatusInternalServerError, "api_error", "streaming unsupported")
		return
	}
	setSSEHeaders(w)
	sw := newSSEWriter(w, flusher)
	stopHB := sw.startHeartbeat(sseHeartbeat)
	defer stopHB()

	start := messagesResponse{
		ID:         newIDSep("msg", "_"),
		Type:       "message",
		Role:       "assistant",
		Model:      modelID,
		Content:    []messagesContentBlock{},
		StopReason: "",
	}
	sw.writeEvent("message_start", map[string]any{
		"type": "message_start", "message": start,
	})
	sw.writeEvent("content_block_start", map[string]any{
		"type": "content_block_start", "index": 0,
		"content_block": messagesContentBlock{Type: "text", Text: ""},
	})

	var b strings.Builder
	usage, genErr := s.runStream(ctx, modelID, req, func(text string) {
		b.WriteString(text)
		sw.writeEvent("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "text_delta", "text": text},
		})
	})
	if genErr != nil {
		// Anthropic terminates a failed stream with an error event; no
		// message_stop follows. A context-length refusal is a client error.
		stopHB()
		typ := "api_error"
		if errors.Is(genErr, engine.ErrContextLength) {
			typ = "invalid_request_error"
		}
		sw.writeEvent("error", anthropicErrorBody{
			Type:  "error",
			Error: anthropicErrorDetail{Type: typ, Message: "generation failed: " + genErr.Error()},
		})
		return
	}
	stopHB()

	sw.writeEvent("content_block_stop", map[string]any{
		"type": "content_block_stop", "index": 0,
	})
	final := newMessagesResponse(modelID, b.String(), usage, req.MaxTokens)
	sw.writeEvent("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": final.StopReason, "stop_sequence": nil},
		"usage": messagesUsage{InputTokens: usage.PromptTokens, OutputTokens: usage.CompletionTokens},
	})
	sw.writeEvent("message_stop", map[string]any{"type": "message_stop"})
}
