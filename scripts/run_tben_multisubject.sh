#!/usr/bin/env bash
# TBEn readout controls across NSD subjects 1, 2, 5, 7.
#
# Phase 1 runs Variant A on every subject first (~30 min each), because that is
# the cheapest way to see whether the one-stage result on subj07 holds up.
# Phase 2 then runs Variant B, which is the decisive contrast but costs hours.
# Anything already marked complete is skipped, so this is safe to re-run.
set -euo pipefail

PYTHON=${PYTHON:-.venv/bin/python}
SUBJECTS=${SUBJECTS:-"7 1 2 5"}
SEED=${SEED:-42}
RESULTS_ROOT=${RESULTS_ROOT:-results/wacv_round2}
LOG_ROOT=${LOG_ROOT:-results/wacv_round2/training}
CKPT_ROOT=${CKPT_ROOT:-checkpoints/wacv_round2}
mkdir -p "$RESULTS_ROOT" "$LOG_ROOT"

is_complete() {
  [ -f "$1" ] && "$PYTHON" -c "
import json,sys
try: sys.exit(0 if json.load(open('$1')).get('status')=='complete' else 1)
except Exception: sys.exit(1)
" 2>/dev/null
}

echo "=========== Phase 1: Variant A (one-stage) ==========="
for subj in $SUBJECTS; do
  result="${RESULTS_ROOT}/tben_one_stage_subj${subj}_seed${SEED}.json"
  if is_complete "$result"; then
    echo ">>> subj${subj} Variant A already complete, skipping"
    continue
  fi
  echo ">>> subj${subj} Variant A"
  "$PYTHON" scripts/train_direct_stage2.py \
    --config "configs/train_tben_one_stage_subj${subj}.yaml" \
    --seed "$SEED" \
    --save_dir "${CKPT_ROOT}/tben_one_stage/subj${subj}_seed${SEED}" \
    --result "$result" \
    2>&1 | tee "${LOG_ROOT}/tben_one_stage_subj${subj}_seed${SEED}.log"
done

echo "=========== Phase 2: Variant B (TBEn Stage 1 + our Stage 2) ==========="
for subj in $SUBJECTS; do
  config="configs/train_tben_two_stage_subj${subj}.yaml"
  save_dir="${CKPT_ROOT}/tben_two_stage/subj${subj}_seed${SEED}"

  if [ ! -f "${save_dir}/best_stage1.pth" ]; then
    echo ">>> subj${subj} Variant B Stage 1"
    "$PYTHON" scripts/train_hierarchical_v3.py \
      --config "$config" --stage 1 --seed "$SEED" --save_dir "$save_dir" \
      2>&1 | tee "${LOG_ROOT}/tben_two_stage_s1_subj${subj}_seed${SEED}.log"
  else
    echo ">>> subj${subj} Variant B Stage 1 already present, skipping"
  fi

  if [ ! -f "${save_dir}/best_model.pth" ]; then
    echo ">>> subj${subj} Variant B Stage 2"
    "$PYTHON" scripts/train_hierarchical_v3.py \
      --config "$config" --stage 2 --seed "$SEED" \
      --load_stage1 "${save_dir}/best_stage1.pth" --save_dir "$save_dir" \
      2>&1 | tee "${LOG_ROOT}/tben_two_stage_s2_subj${subj}_seed${SEED}.log"
  else
    echo ">>> subj${subj} Variant B Stage 2 already present, skipping"
  fi
done

echo "=========== Done ==========="
