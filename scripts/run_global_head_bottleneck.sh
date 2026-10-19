#!/usr/bin/env bash
# Bottleneck global head (128 queries kept, 49152 -> 512 -> 14278): 282M -> 54.6M total.
# Same Stage 1 checkpoint and Stage-2-only protocol as run_global_head_slim.sh,
# so global_hidden is the only difference from the published model.
set -uo pipefail
L=results/wacv_round2/training
tag=q128h512
d="checkpoints/wacv_round2/global_head_slim/${tag}"

# Wait for the running slim queue to release the GPU.
while pgrep -f "run_global_head_slim.sh" > /dev/null; do sleep 60; done

echo ">>> ${tag} start $(date +%H:%M)"
.venv/bin/python scripts/train_hierarchical_v3.py \
  --config "configs/slim_${tag}_subj7.yaml" --stage 2 --seed 42 \
  --load_stage1 checkpoints/wacv_round2/global_head_slim/stage1_subj7.pth \
  --save_dir "$d" > "${L}/slim_${tag}_subj7.log" 2>&1
best=$(grep -a "Saved best Stage 2" "${L}/slim_${tag}_subj7.log" | tail -1 | grep -o "[0-9.]*$")
echo ">>> ${tag} done best=${best} $(date +%H:%M)"
