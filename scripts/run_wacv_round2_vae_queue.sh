#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/train_hierarchical_v3_huwng.yaml}
STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-checkpoints/chasm_rerun_s7/best_stage1.pth}
PYTHON=${PYTHON:-.venv/bin/python}
RESULTS_ROOT=${RESULTS_ROOT:-results/wacv_round2/training}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/wacv_round2}

mkdir -p "$RESULTS_ROOT" "$OUTPUT_ROOT"

# Keep the single GPU exclusive: wait for the official direct baseline first.
while pgrep -f "[s]cripts/train_direct_stage2.py.*direct_stage2_only_subj7_seed42_bf16_lr1e5.json" >/dev/null; do
  sleep 30
done

run_variant() {
  local name=$1
  local vae_mode=$2
  local beta=$3
  local log_path="${RESULTS_ROOT}/${name}_subj7_seed42.log"

  if [[ -f "$log_path" ]] && rg -q "Training Complete" "$log_path"; then
    echo "Skipping completed run: $name"
    return
  fi

  /usr/bin/time -v "$PYTHON" -u scripts/train_hierarchical_v3.py \
    --config "$CONFIG" \
    --stage 2 \
    --load_stage1 "$STAGE1_CHECKPOINT" \
    --guidance_mode predicted \
    --vae_mode "$vae_mode" \
    --kl_weight "$beta" \
    --seed 42 \
    --save_dir "${OUTPUT_ROOT}/${name}/seed_42" \
    2>&1 | tee "$log_path"
}

run_variant vae_deterministic deterministic 0
run_variant vae_beta0 stochastic 0

echo "WACV round-2 VAE queue complete."
