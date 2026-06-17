#!/usr/bin/env bash
# Full fit pipeline: NLL → BFGS optimization → results → plots
set -e

CONFIG="${1:-config_amp.yml}"
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
    --fit --maxiter 1 \
    --backend cuda_v3 \
    --init /home/jiangy/ana/test_4pi/test_amp/pw_cfit5_td6_fix29/final_params_0.json \
    --save "$PREFIX/results.json" \
    --plot "$PREFIX/plots/" \
    --fix-mass-width

echo ""
echo "=== done ==="
echo "results: $PREFIX/results.json"
echo "plots:   $PREFIX/plots/"

echo "=== fitting ==="
python3 run_fit.py \
    --config config_amp.yml \
    --data data/data_arrays.npz \
    --phsp data/phsp_arrays.npz \
    --fit --maxiter 200 \
    --backend cuda_v3 \
    --init /home/jiangy/ana/test_4pi/test_amp/pw_cfit5_td6_fix29/final_params_0.json \
    --fix-mass-width \
    --save "$PREFIX/results.json" \
    --plot "$PREFIX/plots/"
