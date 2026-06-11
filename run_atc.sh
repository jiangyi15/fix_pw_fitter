source /home/jiangy/miniconda3/envs/cann/Ascend/ascend-toolkit/set_env.sh
atc --model pwa_forward_norm.onnx --framework 5 --soc_version Ascend910B \
    --output pwa_forward_model \
    --fusion_switch_file fusion_switch.cfg
