#!/usr/bin/env bash
# Does CHASMBrain need a 260M-parameter global head? Stage 2 only, subj7,
# reusing the published Stage 1 so only the head size changes.
set -uo pipefail
L=results/wacv_round2/training
for tag in q32h1024 q16h512; do
  d="checkpoints/wacv_round2/global_head_slim/${tag}"
  echo ">>> ${tag} start $(date +%H:%M)"
  .venv/bin/python scripts/train_hierarchical_v3.py \
    --config "configs/slim_${tag}_subj7.yaml" --stage 2 --seed 42 \
    --load_stage1 checkpoints/wacv_round2/global_head_slim/stage1_subj7.pth \
    --save_dir "$d" > "${L}/slim_${tag}_subj7.log" 2>&1
  best=$(grep -a "Saved best Stage 2" "${L}/slim_${tag}_subj7.log" | tail -1 | grep -o "[0-9.]*$")
  echo ">>> ${tag} done best=${best} $(date +%H:%M)"
done
echo "=== SLIM DONE $(date +%H:%M) ==="
