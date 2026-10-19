#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/train_hierarchical_v3_huwng.yaml}
STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-checkpoints/chasm_rerun_s7/best_stage1.pth}
SEEDS=${SEEDS:-"42"}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/wacv_round2}
RESULTS_ROOT=${RESULTS_ROOT:-results/wacv_round2/training}
PYTHON=${PYTHON:-.venv/bin/python}

run_variant() {
  local name=$1
  local guidance=$2
  local vae=$3
  local beta=$4
  local seed=$5
  local output_dir="${OUTPUT_ROOT}/${name}/seed_${seed}"
  mkdir -p "$(dirname "$output_dir")"
  mkdir -p "$RESULTS_ROOT"

  /usr/bin/time -v "$PYTHON" scripts/train_hierarchical_v3.py \
    --config "$CONFIG" \
    --stage 2 \
    --load_stage1 "$STAGE1_CHECKPOINT" \
    --guidance_mode "$guidance" \
    --vae_mode "$vae" \
    --kl_weight "$beta" \
    --seed "$seed" \
    --save_dir "$output_dir" \
    2>&1 | tee "${RESULTS_ROOT}/${name}_seed${seed}.log"
}

for seed in $SEEDS; do
  # Cluster-conditioning ablation; all other settings are matched.
  run_variant cluster_full predicted stochastic 0.001 "$seed"
  run_variant cluster_none none stochastic 0.001 "$seed"
  run_variant cluster_shuffle shuffle stochastic 0.001 "$seed"
  run_variant cluster_oracle oracle stochastic 0.001 "$seed"

  # VAE ablation under the full predicted-guidance setting.
  run_variant vae_deterministic predicted deterministic 0 "$seed"
  run_variant vae_beta0 predicted stochastic 0 "$seed"
done
