#!/usr/bin/env bash
# Variant B for one subject: TBEn readout as Stage 1, then our unchanged Stage 2.
set -euo pipefail
subj=$1; seed=${2:-42}
cfg="configs/train_tben_two_stage_subj${subj}.yaml"
dir="checkpoints/wacv_round2/tben_two_stage/subj${subj}_seed${seed}"
log="results/wacv_round2/training"

[ -f "${dir}/best_stage1.pth" ] || \
  .venv/bin/python scripts/train_hierarchical_v3.py --config "$cfg" --stage 1 \
    --seed "$seed" --save_dir "$dir" > "${log}/tben_B_s1_subj${subj}.log" 2>&1

.venv/bin/python scripts/train_hierarchical_v3.py --config "$cfg" --stage 2 \
  --seed "$seed" --load_stage1 "${dir}/best_stage1.pth" --save_dir "$dir" \
  > "${log}/tben_B_s2_subj${subj}.log" 2>&1
