#!/bin/bash
# Sequential run: 1->7 (adapt), 12->7 (full), 15->7 (full), 125->7 (full)
# Early stopping: patience=20 (stage1/2) đã có sẵn trong train_exp2.py
set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$(pwd):${PYTHONPATH}"

echo "======================================"
echo "[1/4] sub1 -> 7  (ADAPT only, reuse 1-7 phase1)"
echo "======================================"
python rebutal/exp2/scripts/adapt_subj5.py \
    --config rebutal/exp2/configs/exp2_1-7.yaml \
    --checkpoint rebutal/exp2/checkpoints/1-7/phase1/best_stage2.pth

echo ""
echo "======================================"
echo "[2/4] sub12 -> 7  (FULL: stage1 + stage2 + adapt)"
echo "======================================"
python rebutal/exp2/scripts/train_exp2.py \
    --config rebutal/exp2/configs/exp2_12-7.yaml --stage 1
python rebutal/exp2/scripts/train_exp2.py \
    --config rebutal/exp2/configs/exp2_12-7.yaml --stage 2 \
    --load_stage1 rebutal/exp2/checkpoints/12-7/phase1/best_stage1.pth
python rebutal/exp2/scripts/adapt_subj5.py \
    --config rebutal/exp2/configs/exp2_12-7.yaml \
    --checkpoint rebutal/exp2/checkpoints/12-7/phase1/best_stage2.pth

echo ""
echo "======================================"
echo "[3/4] sub15 -> 7  (FULL: stage1 + stage2 + adapt)"
echo "======================================"
python rebutal/exp2/scripts/train_exp2.py \
    --config rebutal/exp2/configs/exp2_15-7.yaml --stage 1
python rebutal/exp2/scripts/train_exp2.py \
    --config rebutal/exp2/configs/exp2_15-7.yaml --stage 2 \
    --load_stage1 rebutal/exp2/checkpoints/15-7/phase1/best_stage1.pth
python rebutal/exp2/scripts/adapt_subj5.py \
    --config rebutal/exp2/configs/exp2_15-7.yaml \
    --checkpoint rebutal/exp2/checkpoints/15-7/phase1/best_stage2.pth

echo ""
echo "======================================"
echo "[4/4] sub125 -> 7  (FULL: stage1 + stage2 + adapt)"
echo "======================================"
python rebutal/exp2/scripts/train_exp2.py \
    --config rebutal/exp2/configs/exp2_125-7.yaml --stage 1
python rebutal/exp2/scripts/train_exp2.py \
    --config rebutal/exp2/configs/exp2_125-7.yaml --stage 2 \
    --load_stage1 rebutal/exp2/checkpoints/125-7/phase1/best_stage1.pth
python rebutal/exp2/scripts/adapt_subj5.py \
    --config rebutal/exp2/configs/exp2_125-7.yaml \
    --checkpoint rebutal/exp2/checkpoints/125-7/phase1/best_stage2.pth

echo ""
echo "======================================"
echo "All done. Results: rebutal/exp2/results/exp2_results.json"
echo "======================================"

