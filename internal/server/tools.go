package server

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mgorshkov/tinycoder-openai/internal/engine"
)

// This file implements OpenAI-style function/tool calling for the chat
// endpoint. The native engine returns free text, so tool calling works by
// rendering the tools into the system prompt, instructing the model to emit
// calls in a simple <tool_call>{json}</tool_call> format, and parsing that
// format back into structured tool_calls. This is a text-based protocol (not
// native constrained decoding), which keeps it dependency-free and works with
// any instruction-following chat template.

// toolCallMarker closes a tool-call block.
const toolCallMarker = "</tool_call>"

const toolProtocolInstructions = `# Tool use

You can call the functions listed below to complete the user's request.

When you decide to call one or more functions, respond with ONLY the following,
and nothing else (no prose, no markdown, no code fences):

<tool_call>{"name": "<function name>", "arguments": {<arguments as JSON>}}</tool_call>

Emit one <tool_call> block per call and stop immediately after the last block.
The "arguments" value must be valid JSON matching the function's parameters.

If no function call is needed, reply normally in plain text.`

// buildToolInstructions renders the instructions and function list injected into
// the chat system message. It returns "" when there are no tools.
func buildToolInstructions(tools []toolDefinition) string {
	if len(tools) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString(toolProtocolInstructions)
	for _, t := range tools {
		if strings.TrimSpace(t.Function.Name) == "" {
			continue
		}
		b.WriteString("\n\n## ")
		b.WriteString(t.Function.Name)
		if d := strings.TrimSpace(t.Function.Description); d != "" {
			b.WriteString("\n")
			b.WriteString(d)
		}
		if len(t.Function.Parameters) > 0 {
			b.WriteString("\nParameters (JSON Schema): ")
			b.Write(t.Function.Parameters)
		}
	}
	return b.String()
}

// normalizeToolChoice resolves the effective tool_choice: an explicit mode and,
// when a specific function is selected, its name.
func normalizeToolChoice(tc *toolChoice, tools []toolDefinition) (mode, name string) {
	if len(tools) == 0 {
		return "none", ""
	}
	if tc == nil || tc.Mode == "" {
		return "auto", ""
	}
	return tc.Mode, tc.Name
}

// chatContentForEngine flattens a chat message into the text the engine's chat
// template will receive, materializing tool calls and tool results in the
// <tool_call> protocol so multi-turn tool conversations round-trip.
func chatContentForEngine(m chatMessage) string {
	switch m.Role {
	case "tool":
		label := m.ToolCallID
		if label == "" {
			label = m.Name
		}
		if label == "" {
			label = "unknown"
		}
		return "Tool result for " + label + ":\n" + m.Content.text
	case "assistant":
		if len(m.ToolCalls) > 0 {
			var sb strings.Builder
			if t := strings.TrimSpace(m.Content.text); t != "" {
				sb.WriteString(t)
				sb.WriteString("\n")
			}
			for _, tc := range m.ToolCalls {
				sb.WriteString("<tool_call>")
				sb.WriteString(toolCallJSON(tc))
				sb.WriteString("</tool_call>")
			}
			return sb.String()
		}
	}
	return m.Content.text
}

// injectSystem appends instr to the first system message, or prepends a new one.
func injectSystem(msgs *[]engine.Message, instr string) {
	if strings.TrimSpace(instr) == "" {
		return
	}
	for i := range *msgs {
		if (*msgs)[i].Role == "system" {
			base := strings.TrimRight((*msgs)[i].Content, "\n")
			if base != "" {
				base += "\n\n"
			}
			(*msgs)[i].Content = base + instr
			return
		}
	}
	*msgs = append([]engine.Message{{Role: "system", Content: instr}}, *msgs...)
}

// toolCallJSON renders a toolCall back into the {"name","arguments"} object used
// inside a <tool_call> block.
func toolCallJSON(tc toolCall) string {
	args := strings.TrimSpace(string(tc.Function.Arguments))
	if args == "" || args == "null" {
		args = "{}"
	}
	out, err := json.Marshal(struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}{Name: tc.Function.Name, Arguments: json.RawMessage(args)})
	if err != nil {
		return fmt.Sprintf(`{"name":%q,"arguments":{}}`, tc.Function.Name)
	}
	return string(out)
}

// stripToolCalls removes any <tool_call> protocol from text so that, when the
// model emitted prose alongside (or instead of) a well-formed call, the user
// never sees the raw protocol. An unterminated trailing block is dropped too.
func stripToolCalls(text string) string {
	var b strings.Builder
	for {
		start := strings.Index(text, "<tool_call>")
		if start < 0 {
			b.WriteString(text)
			break
		}
		b.WriteString(text[:start])
		rest := text[start+len("<tool_call>"):]
		end := strings.Index(rest, toolCallMarker)
		if end < 0 {
			break // unterminated trailing block: drop the remainder
		}
		text = rest[end+len(toolCallMarker):]
	}
	return strings.TrimSpace(b.String())
}

// parseToolCalls extracts tool calls from a complete model response.
func parseToolCalls(text string) []toolCall {
	calls, _ := parseToolCallsAt(text, false)
	return calls
}

// parseToolCallsAt parses tool calls from text and returns them plus the number
// of bytes consumed (the end of the last complete call). A non-zero consumed
// value means at least one full call was parsed; consumed is used by the
// streaming path to hold back an incomplete trailing block.
func parseToolCallsAt(text string, streaming bool) ([]toolCall, int) {
	_ = streaming
	var (
		calls    []toolCall
		consumed int
	)
	pos := 0
	for {
		rel := strings.Index(text[pos:], "<tool_call>")
		if rel < 0 {
			break
		}
		bodyStart := pos + rel + len("<tool_call>")
		endRel := strings.Index(text[bodyStart:], toolCallMarker)
		if endRel < 0 {
			break // incomplete trailing block
		}
		bodyEnd := bodyStart + endRel
		body := strings.TrimSpace(text[bodyStart:bodyEnd])
		if raw, _, ok := firstCompleteJSON(body); ok {
			if call, ok := callFromJSON(raw); ok {
				calls = append(calls, call)
			}
		}
		pos = bodyEnd + len(toolCallMarker)
		consumed = pos
	}
	if len(calls) == 0 {
		if bare, n, ok := parseBareToolCalls(text); ok {
			return bare, n
		}
	}
	return calls, consumed
}

// parseBareToolCalls accepts models that ignore the delimiters and emit a raw
// {"name","arguments"} object, an array of such objects, or an OpenAI-shaped
// {"tool_calls":[...]}.
func parseBareToolCalls(s string) ([]toolCall, int, bool) {
	raw, end, ok := firstCompleteJSON(s)
	if !ok {
		return nil, 0, false
	}
	if call, ok := callFromJSON(raw); ok {
		return []toolCall{call}, end, true
	}
	var items []json.RawMessage
	if err := json.Unmarshal([]byte(raw), &items); err == nil {
		var calls []toolCall
		for _, it := range items {
			if call, ok := callFromJSON(string(it)); ok {
				calls = append(calls, call)
			}
		}
		if len(calls) > 0 {
			return calls, end, true
		}
	}
	var wrap struct {
		ToolCalls []json.RawMessage `json:"tool_calls"`
	}
	if err := json.Unmarshal([]byte(raw), &wrap); err == nil && len(wrap.ToolCalls) > 0 {
		var calls []toolCall
		for _, it := range wrap.ToolCalls {
			if call, ok := callFromJSON(string(it)); ok {
				calls = append(calls, call)
			}
		}
		if len(calls) > 0 {
			return calls, end, true
		}
	}
	return nil, 0, false
}

// callFromJSON decodes one tool call, accepting either {"name","arguments"} or
// the nested OpenAI shape {"function":{"name","arguments"}}.
func callFromJSON(s string) (toolCall, bool) {
	var probe struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
		Function  *struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		} `json:"function"`
	}
	if err := json.Unmarshal([]byte(s), &probe); err != nil {
		return toolCall{}, false
	}
	name := probe.Name
	args := probe.Arguments
	if probe.Function != nil && probe.Function.Name != "" {
		name = probe.Function.Name
		args = probe.Function.Arguments
	}
	if strings.TrimSpace(name) == "" {
		return toolCall{}, false
	}
	argText := "{}"
	if len(args) > 0 && string(args) != "null" {
		argText = string(args)
	}
	return toolCall{
		ID:       newID("call"),
		Type:     "function",
		Function: functionCall{Name: name, Arguments: argString(argText)},
	}, true
}

// firstCompleteJSON returns the first complete JSON object or array in s, and
// the index just past it.
func firstCompleteJSON(s string) (string, int, bool) {
	for i := 0; i < len(s); i++ {
		if s[i] != '{' && s[i] != '[' {
			continue
		}
		if end, ok := scanJSON(s, i); ok {
			return s[i:end], end, true
		}
		return "", 0, false
	}
	return "", 0, false
}

// scanJSON returns the index just past the balanced JSON value starting at
// start, honoring strings and escapes.
func scanJSON(s string, start int) (int, bool) {
	var stack []byte
	inStr, esc := false, false
	for i := start; i < len(s); i++ {
		c := s[i]
		if inStr {
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{', '[':
			stack = append(stack, c)
		case '}', ']':
			if len(stack) == 0 {
				return 0, false
			}
			open := stack[len(stack)-1]
			if (c == '}' && open != '{') || (c == ']' && open != '[') {
				return 0, false
			}
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				return i + 1, true
			}
		}
	}
	return 0, false
}
