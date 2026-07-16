#!/usr/bin/env bash
# run_atc.sh –  Convert ONNX model → Ascend .om  via ATC.
#
# Prerequisites:
#   python3 create_onnx_model.py --n-events 1000000   # produce fpwfitter_1m.onnx
#   source <ascend-toolkit>/set_env.sh                 # set ATC env
#
# Usage:
#   ./run_atc.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="$SCRIPT_DIR/fpwfitter_1m.onnx"
OUTPUT="$SCRIPT_DIR/fpwfitter_1m"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: $MODEL not found. Run 'python3 create_onnx_model.py --n-events 1000000' first."
    exit 1
fi

echo "=== ATC: ONNX → Ascend ==="
atc --model "$MODEL" \
    --framework 5 \
    --soc_version Ascend910B4 \
    --output "$OUTPUT" \
    --precision_mode force_fp32
echo "=== Done: ${OUTPUT}.om ==="
