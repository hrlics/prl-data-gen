#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "/project/flame/aviralku/envs/prl/bin/activate"

# === local grader config ===
export OPENAI_BASE_URL=http://10.16.0.81:8000/v1
export OPENAI_API_KEY=grader

# === wandb config ===
export WANDB_DIR=/tmp/wandb_logs
export WANDB_CACHE_DIR=/tmp/wandb_cache
export HF_HOME=/tmp/hf_cache # if use this, must export HF_token, otherwise it can't find HF token under ~/.cache/huggingface/token
export HF_TOKEN="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"

timestamp="$(date +'%Y%m%d-%H%M%S')"

cd "${SCRIPT_DIR}"
python -m pipelinerl.launch \
  --config-name=genvf_v9_1_proof_nextN_8B \
  output_dir="/project/flame/aviralku/results/genvf-v9_1-proof-nextN-8B-20260419-121124" \
  llm_grader.name="openai/gpt-oss-20b" \
  world.actor_fraction=4 \
  world.finetune_fraction=4 \
  finetune.rl.entropy_bonus=0.0 \
  finetune.rl.epsilon=1 \
  max_lag=1024

# finetune.rl.epsilon=1 \ is newly added for stability
# output_dir="/project/flame/aviralku/results/genvf-v9_1-proof-nextN-${timestamp}" \