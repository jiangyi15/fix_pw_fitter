#!/usr/bin/env bash
# Full fit pipeline: NLL → BFGS optimization → results → plots
#
# Defaults run the self-contained tutorial (generate the samples first):
#   python tutorials/generate_data.py
#   ./fit.sh
#
# Or override positionally:
#   ./fit.sh <config> <data.npz> <phsp.npz> <output-prefix> [init.json] [maxiter]
set -e

CONFIG="${1:-tutorials/config.yml}"
DATA="${2:-tutorials/data_arr.npz}"
PHSP="${3:-tutorials/phsp_arr.npz}"
PREFIX="${4:-tutorials/fit_output}"
INIT="${5:-tutorials/init_pwa.json}"
MAXITER="${6:-1000}"
BACKEND="${BACKEND:-}"

echo "=== tabpwa full fit ==="
echo "config:  $CONFIG"
echo "data:    $DATA"
echo "phsp:    $PHSP"
echo "init:    $INIT"
echo "backend: ${BACKEND:-<config default>}"
echo "output:  $PREFIX/"
echo ""

mkdir -p "$PREFIX"

INIT_ARG=""
if [ -f "$INIT" ]; then
    INIT_ARG="--init $INIT"
else
    echo "(no init file $INIT — starting from defaults)"
fi

BACKEND_ARG=""
if [ -n "$BACKEND" ]; then
    BACKEND_ARG="--backend $BACKEND"
fi

python run_fit.py \
    --config "$CONFIG" \
    --data "$DATA" \
    --phsp "$PHSP" \
    --fit --maxiter "$MAXITER" \
    $BACKEND_ARG \
    $INIT_ARG \
    --save "$PREFIX/results.json" \
    --plot "$PREFIX/plots/"

echo ""
echo "=== done ==="
echo "results: $PREFIX/results.json"
echo "plots:   $PREFIX/plots/"
