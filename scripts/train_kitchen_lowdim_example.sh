#!/usr/bin/env bash
# Contoh pelatihan FlowPolicyLowdim pada Franka Kitchen melalui Hydra (tanpa random search).
# Jalankan dari direktori yang berisi subfolder FlowPolicy/, misalnya akar repo clone.
#
#   cd FlowPolicy/../   # atau repo root Anda
#   bash scripts/train_kitchen_lowdim_example.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/FlowPolicy"

export PYTHONPATH="$ROOT/FlowPolicy:${PYTHONPATH:-}"
python train.py --config-name=flowpolicy_lowdim.yaml \
    task=kitchen_complete \
    training.seed=42 \
    training.device=cuda:0 \
    checkpoint.save_ckpt=true
