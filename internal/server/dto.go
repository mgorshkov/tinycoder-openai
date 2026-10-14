package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"time"
)

// ---- Models ----

// modelObject is a single entry in the models list, in an OpenAI-compatible
// shape extended with tinycoder metadata.
type modelObject struct {
	ID            string   `json:"id"`
	Object        string   `json:"object"`
	Created       int64    `json:"created"`
	OwnedBy       string   `json:"owned_by"`
	Name          string   `json:"name,omitempty"`
	Architecture  string   `json:"architecture,omitempty"`
	Quantization  string   `json:"quantization,omitempty"`
	ContextLength int      `json:"context_length"`
	SizeBytes     int64    `json:"size_bytes,omitempty"`
	Loaded        bool     `json:"loaded"`
	Capabilities  []string `json:"capabilities,omitempty"`
	Description   string   `json:"description,omitempty"`
}

type modelsResponse struct {
	Object string        `json:"object"`
	Data   []modelObject `json:"data"`
}

// ---- Chat / completions request ----

type chatMessage struct {
	Role    string         `json:"role"`
	Content messageContent `json:"content"`
}

// messageContent is a message's content, which OpenAI-compatible clients send
// either as a plain string or as an array of typed content parts, e.g.
// [{"type":"text","text":"hello"}]. It unmarshals both shapes and renders to
// plain text, so multimodal-shaped requests still generate a completion.
type messageContent struct {
	text string
}

// contentPart is one entry of an array-form content value. Only text parts
// carry promotable data; other part types (images, etc.) are ignored.
type contentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

func (c *messageContent) UnmarshalJSON(b []byte) error {
	b = bytes.TrimSpace(b)
	if len(b) == 0 || string(b) == "null" {
		c.text = ""
		return nil
	}
	switch b[0] {
	case '"':
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		c.text = s
		return nil
	case '[':
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
		c.text = sb.String()
		return nil
	default:
		return errors.New("message content must be a string or an array of parts")
	}
}

func (c messageContent) MarshalJSON() ([]byte, error) {
	return json.Marshal(c.text)
}

type generationRequest struct {
	Model  string        `json:"model"`
	Prompt string        `json:"prompt,omitempty"`
	Input  string        `json:"input,omitempty"`
	Stream bool          `json:"stream,omitempty"`
	Chat   []chatMessage `json:"messages,omitempty"`

	MaxTokens     *int     `json:"max_tokens,omitempty"`
	Temperature   *float32 `json:"temperature,omitempty"`
	TopP          *float32 `json:"top_p,omitempty"`
	TopK          *float32 `json:"top_k,omitempty"`
	RepeatPenalty *float32 `json:"repeat_penalty,omitempty"`
	RepeatLastN   *int     `json:"repeat_last_n,omitempty"`
	Seed          *uint32  `json:"seed,omitempty"`
}

func (r generationRequest) promptText() string {
	if r.Prompt != "" {
		return r.Prompt
	}
	return r.Input
}

// ---- Responses ----

type usageDTO struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type completionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []completionChoice `json:"choices"`
	Usage   usageDTO           `json:"usage"`
}

type completionChoice struct {
	Index        int    `json:"index"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason"`
}

type chatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   usageDTO     `json:"usage"`
}

type chatChoice struct {
	Index        int         `json:"index"`
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// ---- Streaming chunks ----

type completionChunk struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []completionChoice `json:"choices"`
}

type delta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type chatChunk struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []chatChoiceDelta `json:"choices"`
}

type chatChoiceDelta struct {
	Index        int    `json:"index"`
	Delta        delta  `json:"delta"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// ---- Errors ----

type errorBody struct {
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

func nowUnix() int64 { return time.Now().Unix() }

// errorTypeFor maps an error to an OpenAI-style type string.
func errorTypeFor(err error) string {
	switch {
	case errors.Is(err, ErrModelNotFound):
		return "invalid_request_error"
	default:
		return "server_error"
	}
}
