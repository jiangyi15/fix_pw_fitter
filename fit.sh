#!/usr/bin/env bash
# Full fit pipeline: NLL → BFGS optimization → results → plots
set -e

CONFIG="${1:-config_angle.yml}"
DATA="${2:-data/data_arrays.npz}"
PHSP="${3:-data/phsp_arrays.npz}"
PREFIX="${4:-fit_output}"

echo "=== ampfit full fit ==="
echo "config: $CONFIG"
echo "data:   $DATA"
echo "phsp:   $PHSP"
echo "output: $PREFIX/"
echo ""

mkdir -p "$PREFIX"

python run_fit.py \
    --config "$CONFIG" \
    --data "$DATA" \
    --phsp "$PHSP" \
    --fit --maxiter 1000 \
    --backend cuda \
    --init "$PREFIX/results.json" \
    --save "$PREFIX/results.json" \
    --plot "$PREFIX/plots/" \
    --fix-mass-width

echo ""
echo "=== done ==="
echo "results: $PREFIX/results.json"
echo "plots:   $PREFIX/plots/"
