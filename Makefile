GO ?= go
BIN_DIR := build/bin
BIN := $(BIN_DIR)/tinycoder-openai

.PHONY: all build native native-cuda native-cpu stub test vet fmt run clean

all: build

## build: stub-engine binary (no native deps)
build: stub

## stub: stub-engine binary (no native deps)
stub:
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=0 $(GO) build -o $(BIN) ./cmd/tinycoder-openai

## native: native build (CUDA on by default)
native:
	./scripts/build.sh

## native-cuda: native build with CUDA GPU offload
native-cuda:
	./scripts/build.sh --cuda

## native-cpu: native build, CPU only
native-cpu:
	./scripts/build.sh --cpu

## test: run the Go test suite (stub engine)
test:
	$(GO) test ./...

## vet: static analysis
vet:
	$(GO) vet ./...

## fmt: format Go sources
fmt:
	gofmt -w .

## run: run the stub server on :8080
run: stub
	$(BIN) --listen :8080

## clean: remove build artifacts
clean:
	rm -rf build
