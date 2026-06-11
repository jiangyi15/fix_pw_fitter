source /home/jiangy/miniconda3/envs/cann/Ascend/ascend-toolkit/set_env.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Norm model (sum P*w, no bkg/norm inputs)
atc --model "$SCRIPT_DIR/../pwa_forward_norm.onnx" --framework 5 --soc_version Ascend910B4 \
    --output "$SCRIPT_DIR/pwa_forward_norm_model"

# Forward model (NLL with bkg/norm inputs)
atc --model "$SCRIPT_DIR/../pwa_forward.onnx" --framework 5 --soc_version Ascend910B4 \
    --output "$SCRIPT_DIR/pwa_forward_model"
