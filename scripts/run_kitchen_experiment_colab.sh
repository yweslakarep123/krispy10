#!/usr/bin/env bash
set -euo pipefail

# Colab-friendly launcher for the Franka Kitchen low-dimensional FlowPolicy experiment.
# Example:
#   bash scripts/run_kitchen_experiment_colab.sh
# Optional overrides:
#   N_ITER=100 CV=5 INFER_EPISODES=20 bash scripts/run_kitchen_experiment_colab.sh

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/FlowPolicy:${PYTHONPATH:-}"

python scripts/run_kitchen_experiment.py \
  --seeds 0 42 101 \
  --types with_preprocess no_preprocess \
  --n_iter "${N_ITER:-100}" \
  --cv "${CV:-5}" \
  --proxy_epochs "${PROXY_EPOCHS:-5}" \
  --proxy_steps_per_epoch "${PROXY_STEPS_PER_EPOCH:-50}" \
  --rollout_every "${ROLLOUT_EVERY:-100}" \
  --eval_episodes_during_training "${EVAL_EPISODES:-5}" \
  --inference_episodes "${INFER_EPISODES:-20}" \
  --cache_path "${CACHE_PATH:-data/kitchen_complete_v2_episodes.npz}" \
  --output_root "${OUTPUT_ROOT:-results_kitchen}"
