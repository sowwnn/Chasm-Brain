#!/usr/bin/env bash
# TBEn Stage 1 + TBEn Stage 2. Stage 1 checkpoints are reused from the
# converged tben_two_stage run, so only Stage 2 trains here and the two
# variants differ in exactly one thing: the Stage 2 backbone.
set -uo pipefail
L=results/wacv_round2/training
for s in 7 5 1 2; do
  d="checkpoints/wacv_round2/tben_both/subj${s}_seed42"
  echo ">>> subj${s} start $(date +%H:%M)"
  .venv/bin/python scripts/train_hierarchical_v3.py \
    --config "configs/train_tben_both_subj${s}.yaml" --stage 2 --seed 42 \
    --load_stage1 "${d}/best_stage1.pth" --save_dir "$d" \
    > "${L}/tben_both_s2_subj${s}.log" 2>&1
  rc=$?
  best=$(grep -a "Saved best Stage 2" "${L}/tben_both_s2_subj${s}.log" | tail -1 | grep -o "[0-9.]*$")
  echo ">>> subj${s} done rc=${rc} best=${best} $(date +%H:%M)"
done
echo "=== TBEN_BOTH DONE $(date +%H:%M) ==="
