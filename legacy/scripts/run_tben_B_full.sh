#!/usr/bin/env bash
# Variant B, done properly: Stage 1 to early-stopping first, then Stage 2.
set -uo pipefail
L=results/wacv_round2/training

echo "=== Stage 1: 4 subjects in parallel, to early stopping ==="
for s in 1 2 5 7; do
  d="checkpoints/wacv_round2/tben_two_stage/subj${s}_seed42"
  .venv/bin/python scripts/train_hierarchical_v3.py \
    --config "configs/train_tben_two_stage_subj${s}.yaml" --stage 1 --seed 42 \
    --save_dir "$d" > "${L}/tben_B_s1_subj${s}.log" 2>&1 &
done
wait
echo "=== Stage 1 done $(date +%H:%M) ==="

echo "=== Stage 2: two lanes, two jobs at a time ==="
( for s in 5 1; do
    d="checkpoints/wacv_round2/tben_two_stage/subj${s}_seed42"
    .venv/bin/python scripts/train_hierarchical_v3.py \
      --config "configs/train_tben_two_stage_subj${s}.yaml" --stage 2 --seed 42 \
      --load_stage1 "${d}/best_stage1.pth" --save_dir "$d" \
      > "${L}/tben_B_s2_subj${s}.log" 2>&1
  done ) &
( for s in 7 2; do
    d="checkpoints/wacv_round2/tben_two_stage/subj${s}_seed42"
    .venv/bin/python scripts/train_hierarchical_v3.py \
      --config "configs/train_tben_two_stage_subj${s}.yaml" --stage 2 --seed 42 \
      --load_stage1 "${d}/best_stage1.pth" --save_dir "$d" \
      > "${L}/tben_B_s2_subj${s}.log" 2>&1
  done ) &
wait
echo "=== ALL DONE $(date +%H:%M) ==="
