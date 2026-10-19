#!/usr/bin/env bash
# Bang 2: TBEn Stage 1 (converged checkpoint) + Mamba Stage 2. Two lanes.
set -uo pipefail
L=results/wacv_round2/training
run() {
  for s in "$@"; do
    d="checkpoints/wacv_round2/tben_two_stage/subj${s}_seed42"
    echo ">>> subj${s} Stage 2 start $(date +%H:%M)"
    .venv/bin/python scripts/train_hierarchical_v3.py \
      --config "configs/train_tben_two_stage_subj${s}.yaml" --stage 2 --seed 42 \
      --load_stage1 "${d}/best_stage1.pth" --save_dir "$d" \
      > "${L}/tben_B_s2_subj${s}.log" 2>&1
    echo ">>> subj${s} Stage 2 done rc=$? $(date +%H:%M)"
  done
}
run 5 1 &
run 7 2 &
wait
echo "=== BANG 2 DONE $(date +%H:%M) ==="
