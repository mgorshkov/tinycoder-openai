#!/usr/bin/env bash
#
# build.sh — build the tinycoder-openai provider.
#
#   ./scripts/build.sh                 # native CUDA build (default)
#   ./scripts/build.sh --cpu           # native build, CPU only
#   ./scripts/build.sh --stub          # stub engine (no native deps; dev/tests)
#   ./scripts/build.sh --debug         # Debug build
#
# The native build fetches tinycoder-inference from git via CMake FetchContent
# (like the engine fetches its own np/cmake dependencies), and produces
# build/native/libtinycoder_bridge.so, then compiles the Go binary with CGO
# (the native engine is the default; no build tag is needed). Pin a revision
# with `-DTINYCODER_INFERENCE_TAG=<tag>` passed via CMAKE_ARGS, e.g.
#   CMAKE_ARGS="-DTINYCODER_INFERENCE_TAG=v1.2" ./scripts/build.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="native"
CUDA="ON"
BUILD_TYPE="Release"
# Extra CMake arguments, e.g. CMAKE_ARGS="-DTINYCODER_INFERENCE_TAG=v1.2".
read -r -a EXTRA_CMAKE_ARGS <<<"${CMAKE_ARGS:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --native) MODE="native" ;;
        --stub)   MODE="stub" ;;
        --cuda)   CUDA="ON" ;;
        --cpu)    CUDA="OFF" ;;
        --debug)  BUILD_TYPE="Debug" ;;
        -h|--help)
            sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

mkdir -p build/bin

if [[ "$MODE" == "native" ]]; then
    echo "==> Configuring native bridge (CUDA=$CUDA, type=$BUILD_TYPE)"
    # AVX-512 (incl. VBMI/VBMI2)/AMX stay OFF by default (portable AVX2 build);
    # a base engine compiled with those ISA extensions SIGILLs on hosts that
    # lack them. Enable explicitly with -DENABLE_AVX512=ON (and
    # -DENABLE_AVX512_VBMI=ON / -DENABLE_AVX512_VBMI2=ON) only on known-capable
    # hosts. CUDA is ON by default (pass --cpu to disable).
    cmake -B build/native -S . \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DENABLE_CUDA="$CUDA" \
        -DENABLE_AVX2=ON \
        -DENABLE_AVX512=OFF \
        -DENABLE_AVX512_VBMI=OFF \
        -DENABLE_AVX512_VBMI2=OFF \
        -DENABLE_AMX=OFF \
        "${EXTRA_CMAKE_ARGS[@]}"
    echo "==> Building tinycoder_bridge"
    cmake --build build/native --config "$BUILD_TYPE" -j"$(nproc)"
    echo "==> Building Go binary (native engine)"
    CGO_ENABLED=1 go build \
        -o build/bin/tinycoder-openai ./cmd/tinycoder-openai
else
    echo "==> Building Go binary (stub engine)"
    CGO_ENABLED=0 go build -tags tinycoder_stub \
        -o build/bin/tinycoder-openai ./cmd/tinycoder-openai
fi

echo "==> Done: build/bin/tinycoder-openai"
