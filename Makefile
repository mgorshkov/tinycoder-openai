GO ?= go
BIN_DIR := build/bin
BIN := $(BIN_DIR)/tinycoder-openai

.PHONY: all build native native-cuda native-cpu stub test vet fmt run clean deploy deploy-dry

all: build

## build: native-engine binary (CUDA on by default)
build: native

## stub: stub-engine binary (no native deps; dev/tests)
stub:
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=0 $(GO) build -tags tinycoder_stub -o $(BIN) ./cmd/tinycoder-openai

## native: native build (CUDA on by default)
native:
	./scripts/build.sh

## native-cuda: native build with CUDA GPU offload
native-cuda:
	./scripts/build.sh --cuda

## native-cpu: native build, CPU only
native-cpu:
	./scripts/build.sh --cpu

## test: run the Go test suite (stub engine; no native toolchain needed)
test:
	$(GO) test -tags tinycoder_stub ./...

## vet: static analysis (stub engine; no native toolchain needed)
vet:
	$(GO) vet -tags tinycoder_stub ./...

## fmt: format Go sources
fmt:
	gofmt -w .

## run: run the native server on :8080
run: native
	$(BIN) --listen :8080

## deploy: install build artifacts to the paths in deploy/deploy.config.json (needs root)
deploy:
	./scripts/deploy.sh

## deploy-dry: show what deploy would do without changing anything
deploy-dry:
	./scripts/deploy.sh --dry-run

## clean: remove build artifacts
clean:
	rm -rf build
