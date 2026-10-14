cmake_minimum_required(VERSION 3.18.0)

# TinyCoder OpenAI Provider — native bridge build
# ---------------------------------------------------------------------------
# Builds the C ABI shim (native/tinycoder_shim.cpp) into a shared library
# (libtinycoder_bridge.so) linked against the tinycoder-inference engine. The
# Go binary is then built with:
#
#   CGO_ENABLED=1 go build -tags tinycoder_native \
#       -o build/bin/tinycoder-openai ./cmd/tinycoder-openai
#
# Use scripts/build.sh to automate both steps.
#
# The inference engine is fetched from git via FetchContent (like np/cmake in
# the engine itself). Use -DTINYCODER_INFERENCE_TAG=<tag> to pin a revision.
# ---------------------------------------------------------------------------
project(tinycoder_openai CXX)

include(FetchContent)

# ---- Options ----
# CUDA GPU offload is ON by default; pass -DENABLE_CUDA=OFF for a CPU build.
option(ENABLE_CUDA "Build the inference core with CUDA GPU offload" ON)
option(TINYCODER_BUILD_TESTS "Build tinycoder-inference unit tests/benchmarks" OFF)
# Portability defaults for the inference core. AVX-512/AMX are opt-in because a
# binary whose BASE engine code emits those instructions will SIGILL on hosts
# without them (the runtime dispatch only covers the per-file kernel TUs).
# Explicit -DENABLE_AVX512=ON / -DENABLE_AMX=ON still take precedence.
set(ENABLE_AVX2 "Enable AVX2 in the inference core" CACHE BOOL "ON")
set(ENABLE_AVX512 "Enable AVX-512 in the inference core (AVX-512 hosts only)" CACHE BOOL "OFF")
set(ENABLE_AVX512_VBMI "Enable AVX-512 VBMI in the inference core (Ice Lake+ only)" CACHE BOOL "OFF")
set(ENABLE_AVX512_VBMI2 "Enable AVX-512 VBMI2 in the inference core (Ice Lake+ only)" CACHE BOOL "OFF")
set(ENABLE_AMX "Enable AMX in the inference core (Sapphire Rapids+ only)" CACHE BOOL "OFF")

# ---- Inference engine sources ----
# The engine is always fetched from git (FetchContent); local checkouts are
# intentionally unsupported so the built revision is reproducible.
set(TINYCODER_INFERENCE_REPOSITORY "https://github.com/mgorshkov/tinycoder-inference.git"
    CACHE STRING "Git repository for tinycoder-inference")
set(TINYCODER_INFERENCE_TAG "main"
    CACHE STRING "Git tag/branch/commit for tinycoder-inference")

# Forward the engine build options BEFORE MakeAvailable: the engine's own
# FetchContent logic fetches its np/cmake dependencies unless a parent has
# already provided them.
set(BUILD_TESTS ${TINYCODER_BUILD_TESTS} CACHE BOOL "" FORCE)
set(ENABLE_CUDA ${ENABLE_CUDA} CACHE BOOL "" FORCE)
set(ENABLE_AVX2 ${ENABLE_AVX2} CACHE BOOL "" FORCE)
set(ENABLE_AVX512 ${ENABLE_AVX512} CACHE BOOL "" FORCE)
set(ENABLE_AVX512_VBMI ${ENABLE_AVX512_VBMI} CACHE BOOL "" FORCE)
set(ENABLE_AVX512_VBMI2 ${ENABLE_AVX512_VBMI2} CACHE BOOL "" FORCE)
set(ENABLE_AMX ${ENABLE_AMX} CACHE BOOL "" FORCE)

# ---- CUDA device-link setup ----
# tinycoder_core is built with CUDA separable compilation, so its .cu TUs
# reference device-link registration thunks (__cudaRegisterLinkedBinary_*).
# Those are produced by nvcc, but only RESOLVED by nvcc's device-link (-dlink)
# pass, which CMake runs for a target only when the CUDA language is enabled and
# the target opts in (CUDA_SEPARABLE_COMPILATION). The engine's own executables
# get that pass automatically because the engine enables CUDA in its
# Globals.cmake, but tinycoder_bridge is a plain CXX shared library created in
# THIS project — without CUDA enabled here CMake emits no device-link step and
# the .so fails to link with:
#   undefined reference to `__cudaRegisterLinkedBinary_...'
# So enable the CUDA language up front (before the engine is added) and pin the
# architecture list.
#
# The engine hardcodes CMAKE_CUDA_ARCHITECTURES to 75 80 86 in Globals.cmake
# (RTX 20xx/30xx/40xx) with a plain set(), which shadows any parent cache value,
# so the bridge must device-link for the SAME set of architectures or nvlink
# rejects the differing fatbins. Mirror that list here.
if(ENABLE_CUDA)
    # Semicolon-separated list (CMake list); must match the engine's list.
    set(TINYCODER_CUDA_ARCHITECTURES "75;80;86" CACHE STRING
        "CUDA architectures for the bridge device link (must match the engine)")
    set(CMAKE_CUDA_ARCHITECTURES "${TINYCODER_CUDA_ARCHITECTURES}" CACHE STRING
        "CUDA architectures" FORCE)
    enable_language(CUDA)
endif()

# Always clone the engine from git at the requested revision.
message(STATUS "tinycoder-openai: fetching tinycoder-inference from ${TINYCODER_INFERENCE_REPOSITORY}@${TINYCODER_INFERENCE_TAG}")
FetchContent_Declare(
    tinycoder_inference
    GIT_REPOSITORY "${TINYCODER_INFERENCE_REPOSITORY}"
    GIT_TAG "${TINYCODER_INFERENCE_TAG}"
    GIT_SHALLOW TRUE
)

# MakeAvailable clones the engine (git fetch) AND add_subdirectory's it, so we
# must not call add_subdirectory ourselves.
FetchContent_MakeAvailable(tinycoder_inference)
set(TINYCODER_INFERENCE_DIR "${tinycoder_inference_SOURCE_DIR}")

# ---- C ABI bridge shim ----
# Built as a SHARED library so its transitive static dependencies
# (tinycoder_core, np, OpenMP) are all resolved inside it. This makes the cgo
# link line trivial (-ltinycoder_bridge) and avoids static-archive ordering
# problems. The Go cgo directive sets an rpath back to this directory.
add_library(tinycoder_bridge SHARED
    native/tinycoder_shim.cpp
)
set_target_properties(tinycoder_bridge PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    # internal/engine/native -> ../../../build/native
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
target_include_directories(tinycoder_bridge PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/native"
    "${TINYCODER_INFERENCE_DIR}/include"
)
target_compile_features(tinycoder_bridge PRIVATE cxx_std_20)
target_link_libraries(tinycoder_bridge PRIVATE tinycoder_core)

# When CUDA is enabled, make CMake run the device-link step on the bridge so the
# __cudaRegisterLinkedBinary_* thunks pulled from the tinycoder_core static
# library are resolved. cudadevrt provides the device-side runtime those linked
# binaries call into (__nv_cuda_device* helpers) and must be linked explicitly.
if(ENABLE_CUDA)
    set_target_properties(tinycoder_bridge PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    find_library(CUDADEVRT_LIBRARY cudadevrt)
    if(CUDADEVRT_LIBRARY)
        target_link_libraries(tinycoder_bridge PRIVATE "${CUDADEVRT_LIBRARY}")
    else()
        message(WARNING "tinycoder-openai: libcudadevrt not found; the CUDA device link may fail")
    endif()
endif()

message(STATUS "tinycoder-openai: bridge -> ${CMAKE_BINARY_DIR}/libtinycoder_bridge.so")
message(STATUS "tinycoder-openai: CUDA=${ENABLE_CUDA}")
if(ENABLE_CUDA)
    message(STATUS "tinycoder-openai: CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")
endif()
