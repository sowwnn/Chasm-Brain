#!/bin/bash
set -eo pipefail

PYTHON="python"
SCRIPT="scripts/train_hierarchical_v3.py"

CONFIGS=(
    # "configs/exp1c_dinov2.yaml" 
    # "configs/exp1c_clip_sdxl_multi.yaml"
    "configs/exp1c_sam_vit_b_multi.yaml"
    # "configs/exp1c_resnet50.yaml"
    # "configs/exp1c_convnext_large.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
    NAME=$(basename "$CONFIG" .yaml)

    echo "========================================"
    echo "[$NAME] Stage 1 (15 epochs)"
    echo "========================================"
    $PYTHON $SCRIPT --config "$CONFIG" --stage 1 \
        2>&1 | tee "${NAME}_stage1.log"

    echo "[$NAME] Stage 1 done."
done

echo "All Stage 1 complete."
