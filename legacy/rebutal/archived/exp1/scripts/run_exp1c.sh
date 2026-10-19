#!/bin/bash
set -e

PYTHON="python"
SCRIPT="scripts/train_hierarchical_v3.py"
LOG_DIR="logs/exp1c"
mkdir -p "$LOG_DIR"

CONFIGS=(
    "configs/exp1c_dinov2.yaml:checkpoints/exp1c_dinov2"
    "configs/exp1c_clip_vitb16.yaml:checkpoints/exp1c_clip_vitb16"
    "configs/exp1c_clip_sdxl_multi.yaml:checkpoints/exp1c_clip_sdxl_multi"
    "configs/exp1c_sam_vit_b_multi.yaml:checkpoints/exp1c_sam_vit_b_multi"
    "configs/exp1c_resnet50.yaml:checkpoints/exp1c_resnet50"
    "configs/exp1c_convnext_large.yaml:checkpoints/exp1c_convnext_large"
)

for entry in "${CONFIGS[@]}"; do
    CONFIG="${entry%%:*}"
    CKPT_DIR="${entry##*:}"
    NAME=$(basename "$CONFIG" .yaml)

    echo "========================================"
    echo "[$NAME] Stage 1"
    echo "========================================"
    $PYTHON $SCRIPT --config "$CONFIG" --stage 1 \
        2>&1 | tee "$LOG_DIR/${NAME}_stage1.log"

    S1_CKPT="$CKPT_DIR/best_stage1.pth"
    if [ ! -f "$S1_CKPT" ]; then
        echo "ERROR: Stage 1 checkpoint not found at $S1_CKPT, skipping Stage 2."
        continue
    fi

    echo "========================================"
    echo "[$NAME] Stage 2"
    echo "========================================"
    $PYTHON $SCRIPT --config "$CONFIG" --stage 2 --load_stage1 "$S1_CKPT" \
        2>&1 | tee "$LOG_DIR/${NAME}_stage2.log"

    echo "[$NAME] Done."
done

echo "All exp1c experiments complete."
