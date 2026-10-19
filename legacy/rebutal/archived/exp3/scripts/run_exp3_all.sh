#!/usr/bin/env bash
# Exp 3 - Branch ablations (Stage 1 only).
#   Variant B: swap         — what←mean(patches), where←broadcast(cls)   [xsi9 W2, Pq8b W1]
#   Variant C: what_only    — disable where branch                       [Pq8b W6]
#   Variant D: where_only   — disable what branch                        [Pq8b W6]
# 3 variants × 4 subjects (1, 2, 5, 7) = 12 jobs, sequential.
# Reuses exp2/scripts/train_exp2.py (model takes branch_mode from config).

set -euo pipefail

REPO=/home/sowwn/Workspace/ws/2026/I2fMRI
cd "$REPO"

VARIANTS=(swap whatonly whereonly)
SUBJECTS=(1 2 5 7)
TRAIN=rebutal/exp2/scripts/train_exp2.py
LOG_DIR=rebutal/exp3/logs
mkdir -p "$LOG_DIR"

for variant in "${VARIANTS[@]}"; do
  for sid in "${SUBJECTS[@]}"; do
    CFG="rebutal/exp3/configs/exp3_${variant}_subj${sid}.yaml"
    LOG="$LOG_DIR/${variant}_subj${sid}_s1.log"

    echo "=========================================="
    echo "[${variant} | subj${sid}] Stage 1  →  $LOG"
    echo "=========================================="
    python "$TRAIN" --config "$CFG" --stage 1 2>&1 | tee "$LOG"
  done
done

echo ""
echo "All 12 jobs done. Checkpoints: rebutal/exp3/checkpoints/, logs: $LOG_DIR"
