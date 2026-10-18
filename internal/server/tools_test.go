package server

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mgorshkov/tinycoder-openai/internal/engine"
)

func TestParseToolCalls(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want []string // function names
	}{
		{"wrapped", `<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>`, []string{"get_weather"}},
		{"prose then call", "Sure!\n<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>", []string{"a"}},
		{"two calls", "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>", []string{"a", "b"}},
		{"openai nested", `<tool_call>{"function":{"name":"c","arguments":"{\"x\":1}"}}</tool_call>`, []string{"c"}},
		{"bare object", `{"name":"d","arguments":{"y":2}}`, []string{"d"}},
		{"bare openai", `{"tool_calls":[{"function":{"name":"e","arguments":"{}"}}]}`, []string{"e"}},
		{"no call", "just a normal answer", nil},
		// A complete JSON block without the closing marker is tolerated.
		{"missing close marker", `<tool_call>{"name":"f","arguments":{}}`, []string{"f"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			calls := parseToolCalls(tc.in)
			if len(calls) != len(tc.want) {
				t.Fatalf("got %d calls (%+v), want %d", len(calls), calls, len(tc.want))
			}
			for i, name := range tc.want {
				if calls[i].Function.Name != name {
					t.Fatalf("call %d name = %q, want %q", i, calls[i].Function.Name, name)
				}
				if calls[i].ID == "" || calls[i].Type != "function" {
					t.Fatalf("call %d not populated: %+v", i, calls[i])
				}
			}
		})
	}
}

func TestParseToolCallArgumentsJSON(t *testing.T) {
	calls := parseToolCalls(`<tool_call>{"name":"get_weather","arguments":{"city":"Paris","units":"c"}}</tool_call>`)
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	var args map[string]string
	if err := json.Unmarshal([]byte(calls[0].Function.Arguments), &args); err != nil {
		t.Fatalf("arguments not valid JSON: %q: %v", calls[0].Function.Arguments, err)
	}
	if args["city"] != "Paris" || args["units"] != "c" {
		t.Fatalf("unexpected args: %+v", args)
	}
}

func TestStripToolCalls(t *testing.T) {
	in := "Here you go.\n<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\n"
	if got := stripToolCalls(in); got != "Here you go." {
		t.Fatalf("stripToolCalls = %q", got)
	}
	if got := stripToolCalls("no protocol here"); got != "no protocol here" {
		t.Fatalf("stripToolCalls = %q", got)
	}
}

func TestNormalizeToolChoice(t *testing.T) {
	tools := []toolDefinition{{Type: "function", Function: functionDefinition{Name: "a"}}}
	if mode, _ := normalizeToolChoice(nil, nil); mode != "none" {
		t.Fatalf("no tools: mode = %q, want none", mode)
	}
	if mode, _ := normalizeToolChoice(nil, tools); mode != "auto" {
		t.Fatalf("default: mode = %q, want auto", mode)
	}
	if mode, _ := normalizeToolChoice(&toolChoice{Mode: "none"}, tools); mode != "none" {
		t.Fatalf("explicit none: mode = %q", mode)
	}
	if mode, name := normalizeToolChoice(&toolChoice{Mode: "function", Name: "a"}, tools); mode != "function" || name != "a" {
		t.Fatalf("forced: mode=%q name=%q", mode, name)
	}
}

func TestBuildToolInstructions(t *testing.T) {
	instr := buildToolInstructions([]toolDefinition{{
		Type: "function",
		Function: functionDefinition{
			Name:        "get_weather",
			Description: "Look up the weather.",
			Parameters:  json.RawMessage(`{"type":"object"}`),
		},
	}})
	for _, want := range []string{"<tool_call>", "get_weather", "Look up the weather.", `{"type":"object"}`} {
		if !strings.Contains(instr, want) {
			t.Fatalf("instructions missing %q:\n%s", want, instr)
		}
	}
	if buildToolInstructions(nil) != "" {
		t.Fatal("no tools must yield empty instructions")
	}
}

func TestChatContentForEngine(t *testing.T) {
	// tool result
	got := chatContentForEngine(chatMessage{Role: "tool", ToolCallID: "call_1", Content: messageContent{text: "22C"}})
	if !strings.Contains(got, "call_1") || !strings.Contains(got, "22C") {
		t.Fatalf("tool content = %q", got)
	}
	// assistant with tool calls round-trips to the protocol
	got = chatContentForEngine(chatMessage{Role: "assistant", ToolCalls: []toolCall{{
		Function: functionCall{Name: "get_weather", Arguments: `{"city":"Paris"}`},
	}}})
	if !strings.Contains(got, "<tool_call>") || !strings.Contains(got, "get_weather") {
		t.Fatalf("assistant tool content = %q", got)
	}
	// plain message unchanged
	if got := chatContentForEngine(chatMessage{Role: "user", Content: messageContent{text: "hi"}}); got != "hi" {
		t.Fatalf("plain content = %q", got)
	}
}

func TestInjectSystem(t *testing.T) {
	msgs := []engine.Message{{Role: "user", Content: "hi"}}
	injectSystem(&msgs, "INSTR")
	if len(msgs) != 2 || msgs[0].Role != "system" || msgs[0].Content != "INSTR" {
		t.Fatalf("prepend failed: %+v", msgs)
	}
	msgs = []engine.Message{{Role: "system", Content: "base"}, {Role: "user", Content: "hi"}}
	injectSystem(&msgs, "INSTR")
	if msgs[0].Content != "base\n\nINSTR" {
		t.Fatalf("append failed: %q", msgs[0].Content)
	}
	msgs = []engine.Message{{Role: "user", Content: "hi"}}
	injectSystem(&msgs, "")
	if len(msgs) != 1 {
		t.Fatalf("empty instr must be a no-op: %+v", msgs)
	}
}

// toolEngine streams a fixed reply, split into small chunks so the streaming
// tool-call reassembly is exercised.
type toolEngine struct {
	*engine.Stub
	reply string
}

func (e *toolEngine) Generate(ctx context.Context, _ engine.GenerateRequest, onToken engine.TokenFunc) (engine.Usage, error) {
	if !e.Loaded() {
		return engine.Usage{}, engine.ErrNotLoaded
	}
	var u engine.Usage
	u.PromptTokens = 1
	const step = 7
	for i := 0; i < len(e.reply); i += step {
		end := i + step
		if end > len(e.reply) {
			end = len(e.reply)
		}
		select {
		case <-ctx.Done():
			return u, engine.ErrContextCanceled
		default:
		}
		u.CompletionTokens++
		if onToken != nil && !onToken(engine.Token{ID: int32(i), Text: e.reply[i:end]}) {
			break
		}
	}
	u.TotalTokens = u.PromptTokens + u.CompletionTokens
	return u, nil
}

func newToolServer(t *testing.T, reply string) *httptest.Server {
	t.Helper()
	cfg := testConfig() // config.Default() => ToolCalling true
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	mgr := NewManager(cfg, func(id string) engine.Engine {
		return &toolEngine{Stub: engine.NewStub(id), reply: reply}
	}, log)
	ts := httptest.NewServer(New(cfg, mgr, log).Handler())
	t.Cleanup(ts.Close)
	return ts
}

const toolCallReply = `<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>`

func TestChatToolCallsNonStream(t *testing.T) {
	ts := newToolServer(t, toolCallReply)
	body := `{"model":"alpha","messages":[{"role":"user","content":"weather in Paris?"}],
		"tools":[{"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object"}}}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d body=%s", resp.StatusCode, b)
	}
	var out chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if len(out.Choices) != 1 || out.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("finish_reason = %+v", out.Choices)
	}
	tc := out.Choices[0].Message.ToolCalls
	if len(tc) != 1 || tc[0].Function.Name != "get_weather" {
		t.Fatalf("tool_calls = %+v", tc)
	}
	var args map[string]string
	if err := json.Unmarshal([]byte(tc[0].Function.Arguments), &args); err != nil || args["city"] != "Paris" {
		t.Fatalf("arguments = %q err=%v", tc[0].Function.Arguments, err)
	}
}

func TestChatToolCallsStream(t *testing.T) {
	ts := newToolServer(t, toolCallReply)
	body := `{"model":"alpha","messages":[{"role":"user","content":"weather?"}],"stream":true,
		"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type = %q", ct)
	}

	sc := bufio.NewScanner(resp.Body)
	var (
		done         bool
		sawToolCall  bool
		finishReason string
		content      strings.Builder
	)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			done = true
			break
		}
		var chunk chatChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("bad chunk %q: %v", payload, err)
		}
		for _, c := range chunk.Choices {
			if len(c.Delta.ToolCalls) > 0 {
				sawToolCall = true
				if c.Delta.ToolCalls[0].Function.Name != "get_weather" {
					t.Fatalf("unexpected tool call delta: %+v", c.Delta.ToolCalls[0])
				}
			}
			if c.Delta.Content != "" {
				content.WriteString(c.Delta.Content)
			}
			if c.FinishReason != "" {
				finishReason = c.FinishReason
			}
		}
	}
	if !done {
		t.Fatal("stream did not terminate with [DONE]")
	}
	if !sawToolCall {
		t.Fatal("no tool_calls delta observed")
	}
	if finishReason != "tool_calls" {
		t.Fatalf("finish_reason = %q, want tool_calls", finishReason)
	}
	if strings.Contains(content.String(), "<tool_call>") {
		t.Fatalf("raw protocol leaked into content: %q", content.String())
	}
}

// TestChatNoToolsPlainText verifies a normal chat (no tools) is unaffected and
// markup-free replies stream as content.
func TestChatNoToolsPlainText(t *testing.T) {
	ts := newToolServer(t, "plain answer")
	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Choices[0].FinishReason != "stop" || out.Choices[0].Message.Content.text != "plain answer" {
		t.Fatalf("unexpected: %+v", out.Choices[0])
	}
}

// TestChatToolChoiceNone verifies tool_choice:"none" suppresses the protocol and
// passes the model output through unchanged.
func TestChatToolChoiceNone(t *testing.T) {
	ts := newToolServer(t, toolCallReply)
	body := `{"model":"alpha","messages":[{"role":"user","content":"hi"}],"tool_choice":"none",
		"tools":[{"type":"function","function":{"name":"get_weather"}}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if len(out.Choices[0].Message.ToolCalls) != 0 || out.Choices[0].FinishReason != "stop" {
		t.Fatalf("tool_choice:none should suppress parsing: %+v", out.Choices[0])
	}
}

// TestModelsAdvertiseTools verifies the capability string is present.
func TestModelsAdvertiseTools(t *testing.T) {
	ts := newTestServer(t, testConfig())
	resp, err := http.Get(ts.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var list modelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, c := range list.Data[0].Capabilities {
		if c == "tools" {
			found = true
		}
	}
	if !found {
		t.Fatalf("tools capability missing: %+v", list.Data[0].Capabilities)
	}
}
