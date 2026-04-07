#!/usr/bin/env bash
# Build the CUDA shared library
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

NVCC="${NVCC:-nvcc}"
OUT="${SCRIPT_DIR}/libfpwfitter.so"

echo "Building fpwfitter CUDA library..."
"$NVCC" \
    -Xcompiler -fPIC \
    -shared \
    -O3 \
    -arch=native \
    "${SCRIPT_DIR}/fpwfitter.cu" \
    -lcublas \
    -o "$OUT"

echo "→ $OUT"
echo "Done."
