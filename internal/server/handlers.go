package server

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/mgorshkov/tinycoder-openai/internal/engine"
)

// handleModels lists the model registry.
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error", "")
		return
	}
	if !s.authorize(w, r) {
		return
	}
	resp := modelsResponse{Object: "list"}
	resp.Data = s.modelObjects()
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) modelObjects() []modelObject {
	// The config file is authoritative: advertise exactly the declared models.
	reg := s.cfg.Models
	out := make([]modelObject, 0, len(reg))
	for _, m := range reg {
		out = append(out, modelObject{
			ID:            m.ID,
			Object:        "model",
			Created:       nowUnix(),
			OwnedBy:       "tinycoder",
			Name:          m.Name,
			Architecture:  m.Architecture,
			Quantization:  m.Quantization,
			ContextLength: m.ContextLength,
			SizeBytes:     m.SizeBytes,
			Loaded:        s.mgr.IsLoaded(m.ID),
			Capabilities:  s.capabilities(),
			Description:   m.Description,
		})
	}
	return out
}

// capabilities lists the feature strings advertised for a model.
func (s *Server) capabilities() []string {
	caps := []string{"generate", "chat", "streaming"}
	if s.cfg.ToolCalling {
		caps = append(caps, "tools")
	}
	return caps
}

// modelInRegistry reports whether id is declared in the config registry. This is
// the single source of truth for which model ids the API will serve; discovery
// of GGUFs on disk never widens it.
func (s *Server) modelInRegistry(id string) bool {
	for _, m := range s.cfg.Models {
		if m.ID == id {
			return true
		}
	}
	return false
}

// handleGenerate is the tinycoder-native endpoint. It accepts either a raw
// prompt or a messages array.
func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	s.handleCompletion(w, r, false)
}

// handleCompletions is the OpenAI-style text completion endpoint.
func (s *Server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	s.handleCompletion(w, r, false)
}

// handleChatCompletions is the OpenAI-style chat endpoint.
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	s.handleCompletion(w, r, true)
}

func (s *Server) handleCompletion(w http.ResponseWriter, r *http.Request, chatRequired bool) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error", "")
		return
	}
	if !s.authorize(w, r) {
		return
	}

	var req generationRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8<<20))
	// OpenAI-compatible clients send many optional fields we do not model
	// (response_format, frequency_penalty, ...). Unknown fields are ignored
	// rather than rejected so the endpoint stays wire-compatible with clients.
	if err := dec.Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error(), "invalid_request_error", "")
		return
	}

	isChat := chatRequired || len(req.Chat) > 0

	// Tool calling is a chat-only feature. When the client offers tools and does
	// not disable them with tool_choice:"none", a tool protocol is injected into
	// the prompt and calls are parsed back out of the model's text.
	toolInstr := ""
	if isChat && s.cfg.ToolCalling {
		mode, forced := normalizeToolChoice(req.ToolChoice, req.Tools)
		if mode != "none" {
			toolInstr = buildToolInstructions(req.Tools)
			if toolInstr != "" && forced != "" {
				toolInstr += "\n\nFor this request you MUST call the function \"" + forced + "\"."
			}
		}
	}
	parseTools := toolInstr != ""

	genReq, modelID, err := s.buildRequest(req, chatRequired, toolInstr)
	if err != nil {
		code := ""
		status := http.StatusBadRequest
		if errors.Is(err, ErrModelNotFound) {
			code = "model_not_found"
			status = http.StatusNotFound
		}
		writeError(w, status, err.Error(), errorTypeFor(err), code)
		return
	}

	includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage
	if req.Stream {
		if isChat {
			s.streamChat(w, r, modelID, genReq, includeUsage, parseTools)
		} else {
			s.streamCompletion(w, r, modelID, genReq, includeUsage)
		}
		return
	}

	ctx, cancel := s.requestContext(r)
	defer cancel()

	var b strings.Builder
	usage, err := s.mgr.Generate(ctx, modelID, genReq, func(t engine.Token) bool {
		b.WriteString(t.Text)
		return true
	})
	if err != nil {
		s.writeGenerateError(w, err)
		return
	}

	if isChat {
		text := b.String()
		if parseTools {
			if calls := parseToolCalls(text); len(calls) > 0 {
				writeJSON(w, http.StatusOK, chatResponse{
					ID:      newID("chatcmpl"),
					Object:  "chat.completion",
					Created: nowUnix(),
					Model:   modelID,
					Choices: []chatChoice{{
						Index:        0,
						Message:      chatMessage{Role: "assistant", ToolCalls: calls},
						FinishReason: "tool_calls",
					}},
					Usage: toUsageDTO(usage),
				})
				return
			}
			text = stripToolCalls(text)
		}
		writeJSON(w, http.StatusOK, chatResponse{
			ID:      newID("chatcmpl"),
			Object:  "chat.completion",
			Created: nowUnix(),
			Model:   modelID,
			Choices: []chatChoice{{
				Index:        0,
				Message:      chatMessage{Role: "assistant", Content: messageContent{text: text}},
				FinishReason: "stop",
			}},
			Usage: toUsageDTO(usage),
		})
		return
	}
	writeJSON(w, http.StatusOK, completionResponse{
		ID:      newID("cmpl"),
		Object:  "text_completion",
		Created: nowUnix(),
		Model:   modelID,
		Choices: []completionChoice{{
			Index:        0,
			Text:         b.String(),
			FinishReason: "stop",
		}},
		Usage: toUsageDTO(usage),
	})
}

// buildRequest validates the request and resolves it to an engine.GenerateRequest.
func (s *Server) buildRequest(req generationRequest, chatRequired bool, toolInstr string) (engine.GenerateRequest, string, error) {
	modelID := req.Model
	if modelID == "" {
		if len(s.cfg.Models) == 1 {
			modelID = s.cfg.Models[0].ID
		}
	}
	if modelID == "" {
		return engine.GenerateRequest{}, "", errors.New("field 'model' is required")
	}
	// Enforce the config registry on every generation endpoint: a model id that
	// is not declared in the config file is rejected with 404, even if a matching
	// GGUF exists on disk (models_dir is not the serving registry).
	if !s.modelInRegistry(modelID) {
		return engine.GenerateRequest{}, modelID, fmt.Errorf("%w: %s", ErrModelNotFound, modelID)
	}

	out := engine.GenerateRequest{
		MaxTokens:     s.cfg.DefaultMaxTokens,
		Temperature:   0.7,
		TopP:          0.9,
		TopK:          40,
		RepeatPenalty: 1.1,
		RepeatLastN:   64,
	}
	if req.MaxTokens != nil {
		out.MaxTokens = *req.MaxTokens
	}
	if req.Temperature != nil {
		out.Temperature = *req.Temperature
	}
	if req.TopP != nil {
		out.TopP = *req.TopP
	}
	if req.TopK != nil {
		out.TopK = *req.TopK
	}
	if req.RepeatPenalty != nil {
		out.RepeatPenalty = *req.RepeatPenalty
	}
	if req.RepeatLastN != nil {
		out.RepeatLastN = *req.RepeatLastN
	}
	if req.Seed != nil {
		out.Seed = *req.Seed
	}

	if out.MaxTokens < 1 || out.MaxTokens > s.cfg.MaxPromptTokens {
		return out, modelID, fmt.Errorf("max_tokens must be between 1 and %d", s.cfg.MaxPromptTokens)
	}
	if out.Temperature < 0 || out.Temperature > 2 {
		return out, modelID, errors.New("temperature must be between 0 and 2")
	}
	if out.TopP < 0 || out.TopP > 1 {
		return out, modelID, errors.New("top_p must be between 0 and 1")
	}

	if len(req.Chat) > 0 {
		for _, m := range req.Chat {
			out.Messages = append(out.Messages, engine.Message{Role: m.Role, Content: chatContentForEngine(m)})
		}
		injectSystem(&out.Messages, toolInstr)
		return out, modelID, nil
	}
	if chatRequired {
		return out, modelID, errors.New("field 'messages' is required")
	}
	if strings.TrimSpace(req.promptText()) == "" {
		return out, modelID, errors.New("field 'prompt' (or 'input') is required")
	}
	out.Prompt = req.promptText()
	return out, modelID, nil
}

func (s *Server) writeGenerateError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, ErrModelNotFound):
		writeError(w, http.StatusNotFound, err.Error(), "invalid_request_error", "model_not_found")
	case errors.Is(err, ErrCapacity):
		writeError(w, http.StatusServiceUnavailable, err.Error(), "server_error", "capacity")
	case errors.Is(err, engine.ErrContextCanceled):
		// Client went away or timed out; the response is likely already gone.
		writeError(w, http.StatusGatewayTimeout, "generation canceled", "server_error", "timeout")
	case errors.Is(err, engine.ErrNotLoaded):
		writeError(w, http.StatusConflict, "model not loaded", "server_error", "not_loaded")
	default:
		s.log.Error("generation failed", "err", err)
		writeError(w, http.StatusInternalServerError, "generation failed: "+err.Error(), "server_error", "")
	}
}

// ---- streaming ----

func (s *Server) streamCompletion(w http.ResponseWriter, r *http.Request, modelID string, req engine.GenerateRequest, includeUsage bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming unsupported", "server_error", "")
		return
	}
	setSSEHeaders(w)
	id := newID("cmpl")
	s.writeSSE(w, flusher, map[string]any{
		"id": id, "object": "text_completion", "created": nowUnix(), "model": modelID,
		"choices": []map[string]any{{"index": 0, "text": "", "finish_reason": nil}},
	})

	usage := s.runStream(w, r, modelID, req, func(text string) {
		s.writeSSE(w, flusher, completionChunk{
			ID: id, Object: "text_completion", Created: nowUnix(), Model: modelID,
			Choices: []completionChoice{{Index: 0, Text: text, FinishReason: ""}},
		})
	})
	s.writeSSE(w, flusher, completionChunk{
		ID: id, Object: "text_completion", Created: nowUnix(), Model: modelID,
		Choices: []completionChoice{{Index: 0, Text: "", FinishReason: "stop"}},
	})
	if includeUsage {
		u := toUsageDTO(usage)
		s.writeSSE(w, flusher, completionChunk{
			ID: id, Object: "text_completion", Created: nowUnix(), Model: modelID,
			Choices: []completionChoice{}, Usage: &u,
		})
	}
	s.writeSSEDone(w, flusher)
}

func (s *Server) streamChat(w http.ResponseWriter, r *http.Request, modelID string, req engine.GenerateRequest, includeUsage, parseTools bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming unsupported", "server_error", "")
		return
	}
	setSSEHeaders(w)
	id := newID("chatcmpl")
	s.writeSSE(w, flusher, chatChunk{
		ID: id, Object: "chat.completion.chunk", Created: nowUnix(), Model: modelID,
		Choices: []chatChoiceDelta{{Index: 0, Delta: delta{Role: "assistant"}}},
	})

	var (
		acc      strings.Builder
		emitted  int
		callSeen bool
	)
	usage := s.runStream(w, r, modelID, req, func(text string) {
		acc.WriteString(text)
		if !parseTools {
			s.writeSSE(w, flusher, chatChunk{
				ID: id, Object: "chat.completion.chunk", Created: nowUnix(), Model: modelID,
				Choices: []chatChoiceDelta{{Index: 0, Delta: delta{Content: text}}},
			})
			return
		}
		full := acc.String()
		calls, consumed := parseToolCallsAt(full, true)
		if len(calls) > 0 {
			s.emitToolCallDeltas(w, flusher, id, modelID, calls)
			callSeen = true
		}
		end := emitted
		if consumed > 0 {
			// Hold back everything from the first call block onward; only the
			// leading prose (if any) is streamed as content.
			marker := strings.Index(full, "<tool_call>")
			if marker < 0 || marker > len(full) {
				marker = consumed
			}
			end = marker
		} else {
			// No complete call yet: emit all but a possible partial-block tail.
			end = safeContentLen(full)
		}
		if end < emitted {
			end = emitted
		}
		if end > emitted {
			s.writeSSE(w, flusher, chatChunk{
				ID: id, Object: "chat.completion.chunk", Created: nowUnix(), Model: modelID,
				Choices: []chatChoiceDelta{{Index: 0, Delta: delta{Content: full[emitted:end]}}},
			})
			emitted = end
		}
	})

	finish := "stop"
	if parseTools {
		if calls, _ := parseToolCallsAt(acc.String(), false); len(calls) > 0 {
			if !callSeen {
				s.emitToolCallDeltas(w, flusher, id, modelID, calls)
			}
			finish = "tool_calls"
		} else {
			// No tool call: flush any remaining prose held back by the parser.
			rest := stripToolCalls(acc.String())
			if len(rest) > emitted {
				s.writeSSE(w, flusher, chatChunk{
					ID: id, Object: "chat.completion.chunk", Created: nowUnix(), Model: modelID,
					Choices: []chatChoiceDelta{{Index: 0, Delta: delta{Content: rest[emitted:]}}},
				})
			}
		}
	}
	s.writeSSE(w, flusher, chatChunk{
		ID: id, Object: "chat.completion.chunk", Created: nowUnix(), Model: modelID,
		Choices: []chatChoiceDelta{{Index: 0, Delta: delta{}, FinishReason: finish}},
	})
	if includeUsage {
		u := toUsageDTO(usage)
		s.writeSSE(w, flusher, chatChunk{
			ID: id, Object: "chat.completion.chunk", Created: nowUnix(), Model: modelID,
			Choices: []chatChoiceDelta{}, Usage: &u,
		})
	}
	s.writeSSEDone(w, flusher)
}

// emitToolCallDeltas streams tool calls as delta fragments (one chunk per call,
// carrying id, name and the full argument string).
func (s *Server) emitToolCallDeltas(w http.ResponseWriter, flusher http.Flusher, id, modelID string, calls []toolCall) {
	for i, c := range calls {
		s.writeSSE(w, flusher, chatChunk{
			ID: id, Object: "chat.completion.chunk", Created: nowUnix(), Model: modelID,
			Choices: []chatChoiceDelta{{Index: 0, Delta: delta{ToolCalls: []toolCallDelta{{
				Index:    i,
				ID:       c.ID,
				Type:     "function",
				Function: functionCall{Name: c.Function.Name, Arguments: c.Function.Arguments},
			}}}}},
		})
	}
}

// safeContentLen returns the length of the leading prose that is safe to stream
// before any tool-call protocol appears: everything up to a "<tool_call>" marker
// or a trailing partial marker.
func safeContentLen(full string) int {
	idx := strings.Index(full, "<tool_call>")
	if idx < 0 {
		idx = len(full)
	}
	// Hold back a partial "<tool_call>" that may be split across tokens.
	const marker = "<tool_call>"
	maxHold := len(marker) - 1
	if maxHold > idx {
		maxHold = idx
	}
	for hold := maxHold; hold > 0; hold-- {
		if strings.HasSuffix(full[:idx], marker[:hold]) {
			return idx - hold
		}
	}
	return idx
}

// runStream executes generation, forwarding each token through emit and
// returning aggregate usage. If the client disconnects, the context is canceled
// and generation stops; an error is logged only for genuine failures.
func (s *Server) runStream(w http.ResponseWriter, r *http.Request, modelID string, req engine.GenerateRequest, emit func(string)) engine.Usage {
	ctx, cancel := s.requestContext(r)
	defer cancel()
	usage, err := s.mgr.Generate(ctx, modelID, req, func(t engine.Token) bool {
		emit(t.Text)
		return true
	})
	if err != nil {
		s.log.Warn("stream generation ended", "model", modelID, "err", err)
	}
	return usage
}

func setSSEHeaders(w http.ResponseWriter) {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream; charset=utf-8")
	h.Set("Cache-Control", "no-cache, no-transform")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
}

func (s *Server) writeSSE(w http.ResponseWriter, flusher http.Flusher, v any) {
	data, err := json.Marshal(v)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

func (s *Server) writeSSEDone(w http.ResponseWriter, flusher http.Flusher) {
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func toUsageDTO(u engine.Usage) usageDTO {
	return usageDTO{PromptTokens: u.PromptTokens, CompletionTokens: u.CompletionTokens, TotalTokens: u.TotalTokens}
}

// newID returns an id like "chatcmpl-<16 hex chars>" using crypto/rand.
func newID(prefix string) string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return prefix + "-0000000000000000"
	}
	return prefix + "-" + hex.EncodeToString(b[:])
}
