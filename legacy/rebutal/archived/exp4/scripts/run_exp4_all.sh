#!/usr/bin/env bash
# Exp 4 - Top-K% sensitivity analysis (Stage 1 only, subj01).
# Variants: 25%, 50%, 90% (baseline 70% lives in exp2/exp3).
# Addresses xsi9 hyperparameter justification (Priority 3 in plan.md).

set -euo pipefail

REPO=/home/sowwn/Workspace/ws/2026/I2fMRI
cd "$REPO"

VARIANTS=(25 50 90)
TRAIN=rebutal/exp2/scripts/train_exp2.py
LOG_DIR=rebutal/exp4/logs
mkdir -p "$LOG_DIR"

for pct in "${VARIANTS[@]}"; do
  CFG="rebutal/exp4/configs/exp4_topk${pct}_subj1.yaml"
  LOG="$LOG_DIR/topk${pct}_subj1_s1.log"

  echo "=========================================="
  echo "[topk${pct} | subj1] Stage 1  →  $LOG"
  echo "=========================================="
  python "$TRAIN" --config "$CFG" --stage 1 2>&1 | tee "$LOG"
done

echo ""
echo "All 3 jobs done. Checkpoints: rebutal/exp4/checkpoints/, logs: $LOG_DIR"
