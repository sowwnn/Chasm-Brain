#!/bin/bash
set -e

# Ensure project root is in Python path
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Setup name -> config file (Phase 1 train subjects -> Phase 2 adapt subj07)
declare -A SETUPS=(
    ["1-7"]="rebutal/exp2/configs/exp2_1-7.yaml"
    ["12-7"]="rebutal/exp2/configs/exp2_12-7.yaml"
    ["125-7"]="rebutal/exp2/configs/exp2_125-7.yaml"
)

for SETUP in "1-7" "12-7" "125-7"; do
    CONFIG="${SETUPS[$SETUP]}"
    CKPT_DIR="rebutal/exp2/checkpoints/${SETUP}/phase1"

    echo "======================================"
    echo "Exp 2: Cross-Subject  setup=${SETUP}  ->  subj07"
    echo "config: ${CONFIG}"
    echo "======================================"

    # Phase 1 - Stage 1
    echo ""
    echo "[1/3] Phase 1 - Stage 1 (dual-stream backbone)"
    python rebutal/exp2/scripts/train_exp2.py \
        --config "$CONFIG" \
        --stage 1

    # Phase 1 - Stage 2
    echo ""
    echo "[2/3] Phase 1 - Stage 2 (voxel prediction)"
    python rebutal/exp2/scripts/train_exp2.py \
        --config "$CONFIG" \
        --stage 2 \
        --load_stage1 "$CKPT_DIR/best_stage1.pth"

    # Phase 2 - Adapt + Evaluate subj07
    echo ""
    echo "[3/3] Phase 2 - Adapt head to subj07 + Evaluate"
    python rebutal/exp2/scripts/adapt_subj5.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT_DIR/best_stage2.pth"

    echo ""
    echo "Done setup ${SETUP}. Results: rebutal/exp2/results/exp2_results.json"
done

echo ""
echo "======================================"
echo "All setups done."
echo "======================================"
