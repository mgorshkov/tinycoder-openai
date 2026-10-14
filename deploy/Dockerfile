# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build stage: compile the native bridge against tinycoder-inference.
#
# tinycoder-inference is fetched from git by CMake FetchContent, so the build
# context only needs THIS repository:
#   docker build -f deploy/Dockerfile -t tinycoder-openai .
#
# For CUDA, start from an nvidia/cuda image and pass -DENABLE_CUDA=ON
# -DCMAKE_CUDA_ARCHITECTURES=<arch>.
# ---------------------------------------------------------------------------
FROM golang:1.22-bookworm AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake g++ make git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src/tinycoder-openai
COPY . .

ENV CGO_ENABLED=1
RUN cmake -B build/native -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_CUDA=OFF \
        -DENABLE_AVX2=ON \
        -DENABLE_AVX512=OFF \
        -DENABLE_AVX512_VBMI=OFF \
        -DENABLE_AVX512_VBMI2=OFF \
        -DENABLE_AMX=OFF \
    && cmake --build build/native -j"$(nproc)" \
    && go build -o /out/tinycoder-openai ./cmd/tinycoder-openai

# ---------------------------------------------------------------------------
# Runtime stage.
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home tinycoder

WORKDIR /app
COPY --from=build /out/tinycoder-openai /app/tinycoder-openai
COPY --from=build /src/tinycoder-openai/build/native/libtinycoder_bridge.so /app/lib/
COPY config.example.json /app/config.json

ENV LD_LIBRARY_PATH=/app/lib
ENV TINYCODER_LISTEN=:8080
ENV TINYCODER_MODELS_DIR=/data/models

USER tinycoder
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/app/tinycoder-openai", "--version"]

ENTRYPOINT ["/app/tinycoder-openai"]
CMD ["--config", "/app/config.json", "--listen", ":8080"]
