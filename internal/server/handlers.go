package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/mgorshkov/tinycoder-openai/internal/engine"
)

// sseHeartbeat is the interval between keep-alive comment lines written on an
// otherwise idle SSE stream. The inference engine emits no bytes during model
// load and prompt prefill (which can take minutes for large prompts on the
// shared GPU), so without a periodic heartbeat the client's idle/read timeout
// fires before the first token and the stream appears to hang with no output.
//
// 10s is comfortably below the default read timeouts of common clients
// (Node/undici ~30s, most HTTP clients 60s+) while adding negligible traffic.
// It is a var (not a const) so tests can shorten it.
var sseHeartbeat = 10 * time.Second

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
		// Every buildRequest failure is a client-side request problem (unknown
		// model id, missing messages/prompt, out-of-range parameter), so it is
		// typed as an invalid request rather than a server error.
		code := ""
		status := http.StatusBadRequest
		if errors.Is(err, ErrModelNotFound) {
			code = "model_not_found"
			status = http.StatusNotFound
		}
		writeError(w, status, err.Error(), "invalid_request_error", code)
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

	// max_prompt_tokens is a server-side ceiling on generation length, not a
	// hard request-validation bound. OpenAI-compatible clients routinely ask for
	// more than a deployment allows (e.g. Roo Code clamps max_tokens to 20% of
	// the model's declared context window, which exceeds this ceiling for any
	// model above ~163k tokens). Clamp rather than reject so a standard request
	// never fails with a 400: the server quietly generates at most
	// max_prompt_tokens. A request that asks for less than one token is still a
	// client error.
	if out.MaxTokens < 1 {
		return out, modelID, errors.New("max_tokens must be at least 1")
	}
	if out.MaxTokens > s.cfg.MaxPromptTokens {
		out.MaxTokens = s.cfg.MaxPromptTokens
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

// writeGenerateError maps a generation failure to a non-200 HTTP response.
//
// Every branch below writes a status >= 400: cancellation, deadline expiry and
// the no-token guard must never surface as a 200 (a truncation or a hang must
// be observable by the client).
func (s *Server) writeGenerateError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, ErrModelNotFound):
		writeError(w, http.StatusNotFound, err.Error(), "invalid_request_error", "model_not_found")
	case errors.Is(err, ErrCapacity):
		writeError(w, http.StatusServiceUnavailable, err.Error(), "server_error", "capacity")
	case errors.Is(err, engine.ErrNoToken):
		// The engine produced no token within the guard window: report 504
		// rather than hanging the request indefinitely.
		writeError(w, http.StatusGatewayTimeout, err.Error(), "server_error", "no_token")
	case errors.Is(err, engine.ErrContextCanceled),
		errors.Is(err, context.DeadlineExceeded),
		errors.Is(err, context.Canceled):
		// Client went away or timed out; the response may already be gone, but
		// when it is still writable the client must see a timeout status.
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
	sw := newSSEWriter(w, flusher)
	stopHB := sw.startHeartbeat(sseHeartbeat)
	defer stopHB()

	id := newID("cmpl")
	sw.write(map[string]any{
		"id": id, "object": "text_completion", "created": nowUnix(), "model": modelID,
		"choices": []map[string]any{{"index": 0, "text": "", "finish_reason": nil}},
	})

	usage, genErr := s.runStream(w, r, modelID, req, func(text string) {
		sw.write(completionChunk{
			ID: id, Object: "text_completion", Created: nowUnix(), Model: modelID,
			Choices: []completionChoice{{Index: 0, Text: text, FinishReason: ""}},
		})
	})
	if genErr != nil {
		stopHB()
		sw.writeError(genErr)
		sw.writeDone()
		return
	}
	// Stop keep-alives before the terminal events so no comment can be written
	// after [DONE].
	stopHB()
	sw.write(completionChunk{
		ID: id, Object: "text_completion", Created: nowUnix(), Model: modelID,
		Choices: []completionChoice{{Index: 0, Text: "", FinishReason: "stop"}},
	})
	if includeUsage {
		u := toUsageDTO(usage)
		sw.write(completionChunk{
			ID: id, Object: "text_completion", Created: nowUnix(), Model: modelID,
			Choices: []completionChoice{}, Usage: &u,
		})
	}
	sw.writeDone()
}

func (s *Server) streamChat(w http.ResponseWriter, r *http.Request, modelID string, req engine.GenerateRequest, includeUsage, parseTools bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming unsupported", "server_error", "")
		return
	}
	setSSEHeaders(w)
	sw := newSSEWriter(w, flusher)
	stopHB := sw.startHeartbeat(sseHeartbeat)
	defer stopHB()

	id := newID("chatcmpl")
	sw.write(chatChunk{
		ID: id, Object: "chat.completion.chunk", Created: nowUnix(), Model: modelID,
		Choices: []chatChoiceDelta{{Index: 0, Delta: delta{Role: "assistant"}}},
	})

	var (
		acc      strings.Builder
		emitted  int
		callSeen bool
	)
	usage, genErr := s.runStream(w, r, modelID, req, func(text string) {
		acc.WriteString(text)
		if !parseTools {
			sw.write(chatChunk{
				ID: id, Object: "chat.completion.chunk", Created: nowUnix(), Model: modelID,
				Choices: []chatChoiceDelta{{Index: 0, Delta: delta{Content: text}}},
			})
			return
		}
		full := acc.String()
		calls, consumed := parseToolCallsAt(full, true)
		if len(calls) > 0 {
			emitToolCallDeltas(sw, id, modelID, calls)
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
			sw.write(chatChunk{
				ID: id, Object: "chat.completion.chunk", Created: nowUnix(), Model: modelID,
				Choices: []chatChoiceDelta{{Index: 0, Delta: delta{Content: full[emitted:end]}}},
			})
			emitted = end
		}
	})
	if genErr != nil {
		// Surface the failure as an SSE error event instead of a silently
		// truncated stream, then terminate cleanly so the client does not hang.
		stopHB()
		sw.writeError(genErr)
		sw.writeDone()
		return
	}
	// Stop keep-alives before the terminal events so no comment can be written
	// after [DONE].
	stopHB()

	finish := "stop"
	if parseTools {
		if calls, _ := parseToolCallsAt(acc.String(), false); len(calls) > 0 {
			if !callSeen {
				emitToolCallDeltas(sw, id, modelID, calls)
			}
			finish = "tool_calls"
		} else {
			// No tool call: flush any remaining prose held back by the parser.
			rest := stripToolCalls(acc.String())
			if len(rest) > emitted {
				sw.write(chatChunk{
					ID: id, Object: "chat.completion.chunk", Created: nowUnix(), Model: modelID,
					Choices: []chatChoiceDelta{{Index: 0, Delta: delta{Content: rest[emitted:]}}},
				})
			}
		}
	}
	sw.write(chatChunk{
		ID: id, Object: "chat.completion.chunk", Created: nowUnix(), Model: modelID,
		Choices: []chatChoiceDelta{{Index: 0, Delta: delta{}, FinishReason: finish}},
	})
	if includeUsage {
		u := toUsageDTO(usage)
		sw.write(chatChunk{
			ID: id, Object: "chat.completion.chunk", Created: nowUnix(), Model: modelID,
			Choices: []chatChoiceDelta{}, Usage: &u,
		})
	}
	sw.writeDone()
}

// emitToolCallDeltas streams tool calls as delta fragments (one chunk per call,
// carrying id, name and the full argument string).
func emitToolCallDeltas(sw *sseWriter, id, modelID string, calls []toolCall) {
	for i, c := range calls {
		sw.write(chatChunk{
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
// returning aggregate usage plus any terminal error. If the client disconnects,
// the context is canceled and generation stops; the error is logged and returned
// so the caller can terminate the SSE stream instead of hanging.
func (s *Server) runStream(w http.ResponseWriter, r *http.Request, modelID string, req engine.GenerateRequest, emit func(string)) (engine.Usage, error) {
	ctx, cancel := s.requestContext(r)
	defer cancel()
	usage, err := s.mgr.Generate(ctx, modelID, req, func(t engine.Token) bool {
		emit(t.Text)
		return true
	})
	if err != nil {
		s.log.Warn("stream generation ended", "model", modelID, "err", err)
	}
	return usage, err
}

func setSSEHeaders(w http.ResponseWriter) {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream; charset=utf-8")
	h.Set("Cache-Control", "no-cache, no-transform")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
}

// sseWriter serializes writes to a Server-Sent Events stream. It guards against
// concurrent writes so the periodic heartbeat (sent from a separate goroutine
// while the engine is loading or prefilling, producing no tokens yet) can never
// interleave with or race a token write on the same ResponseWriter.
type sseWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	mu      sync.Mutex
}

func newSSEWriter(w http.ResponseWriter, flusher http.Flusher) *sseWriter {
	return &sseWriter{w: w, flusher: flusher}
}

// write emits one SSE data event.
func (s *sseWriter) write(v any) {
	data, err := json.Marshal(v)
	if err != nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	fmt.Fprintf(s.w, "data: %s\n\n", data)
	s.flusher.Flush()
}

// comment emits an SSE comment line (": ..."), which every compliant client
// ignores. It keeps the connection alive during long idle periods.
func (s *sseWriter) comment(text string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	fmt.Fprintf(s.w, ": %s\n\n", text)
	s.flusher.Flush()
}

// writeDone emits the terminating [DONE] sentinel.
func (s *sseWriter) writeDone() {
	s.mu.Lock()
	defer s.mu.Unlock()
	fmt.Fprint(s.w, "data: [DONE]\n\n")
	s.flusher.Flush()
}

// writeError reports a generation failure to the client before the stream ends.
// The payload is an OpenAI-style error object; clients that do not understand it
// simply ignore the event, and the subsequent [DONE] terminates the stream.
func (s *sseWriter) writeError(err error) {
	typ := errorTypeFor(err)
	s.write(errorBody{Error: errorDetail{Message: "generation failed: " + err.Error(), Type: typ}})
}

// startHeartbeat launches a goroutine that emits a keep-alive comment every
// interval until the returned stop function is called. It returns a stop
// function that is safe to call more than once.
func (s *sseWriter) startHeartbeat(interval time.Duration) (stop func()) {
	done := make(chan struct{})
	var once sync.Once
	go func() {
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				s.comment("keep-alive")
			}
		}
	}()
	return func() { once.Do(func() { close(done) }) }
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
