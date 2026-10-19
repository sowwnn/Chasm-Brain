#!/usr/bin/env bash
# Run Variant B Stage 2 for a list of subjects, one after another.
set -uo pipefail
for subj in "$@"; do
  d="checkpoints/wacv_round2/tben_two_stage/subj${subj}_seed42"
  echo ">>> subj${subj} Stage 2 starting $(date +%H:%M)"
  .venv/bin/python scripts/train_hierarchical_v3.py \
    --config "configs/train_tben_two_stage_subj${subj}.yaml" --stage 2 --seed 42 \
    --load_stage1 "${d}/best_stage1.pth" --save_dir "$d" \
    > "results/wacv_round2/training/tben_B_s2_subj${subj}.log" 2>&1
  echo ">>> subj${subj} Stage 2 exited rc=$? $(date +%H:%M)"
done
