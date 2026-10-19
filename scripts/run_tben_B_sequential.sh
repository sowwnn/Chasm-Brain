#!/usr/bin/env bash
# Variant B, fully sequential: one job on the GPU at a time.
# Phase 1 trains every TBEn Stage 1 to early stopping.
# Phase 2 then trains every Stage 2 from those converged checkpoints.
set -uo pipefail
L=results/wacv_round2/training
SUBJECTS="7 5 1 2"

echo "=== PHASE 1: TBEn Stage 1, to early stopping ==="
for s in $SUBJECTS; do
  d="checkpoints/wacv_round2/tben_two_stage/subj${s}_seed42"
  echo ">>> [S1] subj${s} start $(date +%H:%M)"
  .venv/bin/python scripts/train_hierarchical_v3.py \
    --config "configs/train_tben_two_stage_subj${s}.yaml" --stage 1 --seed 42 \
    --save_dir "$d" > "${L}/tben_B_s1_subj${s}.log" 2>&1
  rc=$?
  best=$(grep -a "Saved best Stage 1" "${L}/tben_B_s1_subj${s}.log" | tail -1 | grep -o "[0-9.]*$")
  echo ">>> [S1] subj${s} done rc=${rc} best=${best} $(date +%H:%M)"
done

echo "=== PHASE 2: Mamba Stage 2 from those checkpoints ==="
for s in $SUBJECTS; do
  d="checkpoints/wacv_round2/tben_two_stage/subj${s}_seed42"
  echo ">>> [S2] subj${s} start $(date +%H:%M)"
  .venv/bin/python scripts/train_hierarchical_v3.py \
    --config "configs/train_tben_two_stage_subj${s}.yaml" --stage 2 --seed 42 \
    --load_stage1 "${d}/best_stage1.pth" --save_dir "$d" \
    > "${L}/tben_B_s2_subj${s}.log" 2>&1
  rc=$?
  best=$(grep -a "Saved best Stage 2" "${L}/tben_B_s2_subj${s}.log" | tail -1 | grep -o "[0-9.]*$")
  echo ">>> [S2] subj${s} done rc=${rc} best=${best} $(date +%H:%M)"
done
echo "=== ALL DONE $(date +%H:%M) ==="
