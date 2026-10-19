#!/usr/bin/env bash
# TBEn readout controls for the WACV round-2 rebuttal.
#   Variant A: one-stage structured readout          -> Table 1
#   Variant B: same readout as Stage 1 + our Stage 2 -> Table 4(a)
set -euo pipefail

PYTHON=${PYTHON:-.venv/bin/python}
SEED=${SEED:-42}
RESULTS_ROOT=${RESULTS_ROOT:-results/wacv_round2}
LOG_ROOT=${LOG_ROOT:-results/wacv_round2/training}
CKPT_ROOT=${CKPT_ROOT:-checkpoints/wacv_round2}
mkdir -p "$RESULTS_ROOT" "$LOG_ROOT"

# --- Variant A -------------------------------------------------------------
"$PYTHON" scripts/train_direct_stage2.py \
  --config configs/train_tben_one_stage_subj7.yaml \
  --seed "$SEED" \
  --save_dir "${CKPT_ROOT}/tben_one_stage/subj7_seed${SEED}" \
  --result "${RESULTS_ROOT}/tben_one_stage_subj7_seed${SEED}.json" \
  2>&1 | tee "${LOG_ROOT}/tben_one_stage_seed${SEED}.log"

# --- Variant B, Stage 1 ----------------------------------------------------
"$PYTHON" scripts/train_hierarchical_v3.py \
  --config configs/train_tben_two_stage_subj7.yaml \
  --stage 1 --seed "$SEED" \
  --save_dir "${CKPT_ROOT}/tben_two_stage/seed_${SEED}" \
  2>&1 | tee "${LOG_ROOT}/tben_two_stage_s1_seed${SEED}.log"

# --- Variant B, Stage 2 ----------------------------------------------------
"$PYTHON" scripts/train_hierarchical_v3.py \
  --config configs/train_tben_two_stage_subj7.yaml \
  --stage 2 --seed "$SEED" \
  --load_stage1 "${CKPT_ROOT}/tben_two_stage/seed_${SEED}/best_stage1.pth" \
  --save_dir "${CKPT_ROOT}/tben_two_stage/seed_${SEED}" \
  2>&1 | tee "${LOG_ROOT}/tben_two_stage_s2_seed${SEED}.log"

# --- Paired comparison against CHASMBrain ----------------------------------
"$PYTHON" scripts/evaluate_tben_comparison.py \
  --tben_one_stage_checkpoint "${CKPT_ROOT}/tben_one_stage/subj7_seed${SEED}/best_model.pth" \
  --tben_two_stage_checkpoint "${CKPT_ROOT}/tben_two_stage/seed_${SEED}/best_model.pth" \
  --direct_checkpoint "${CKPT_ROOT}/direct_stage2_only/transformer_subj7_seed42/best_model.pth" \
  --output "${RESULTS_ROOT}/tben_comparison_subj7.json" \
  --per_image_output "${RESULTS_ROOT}/tben_comparison_subj7_per_image.npz" \
  --seed "$SEED" \
  2>&1 | tee "${LOG_ROOT}/tben_comparison_seed${SEED}.log"
