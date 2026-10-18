package server

import (
	"embed"
	"net/http"
	"strings"
)

//go:embed spec/openapi.yaml spec/openapi.json
var specFS embed.FS

// handleSpec serves the OpenAPI 3.1 document as YAML or JSON depending on the
// requested path suffix.
func (s *Server) handleSpec(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error", "")
		return
	}
	name := "spec/openapi.yaml"
	ctype := "application/yaml; charset=utf-8"
	if strings.HasSuffix(r.URL.Path, ".json") {
		name = "spec/openapi.json"
		ctype = "application/json; charset=utf-8"
	}
	data, err := specFS.ReadFile(name)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "spec unavailable", "server_error", "")
		return
	}
	w.Header().Set("Content-Type", ctype)
	w.Header().Set("Cache-Control", "public, max-age=300")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}
