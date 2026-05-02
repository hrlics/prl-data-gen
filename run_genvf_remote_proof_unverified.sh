#!/bin/bash
set -euo pipefail

# sleep 1200

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "/project/flame/aviralku/envs/prl/bin/activate"

export OPENAI_BASE_URL="https://mtllv6vkucczkopr.us-east-2.aws.endpoints.huggingface.cloud/v1"
export OPENAI_API_KEY="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"

# to avoid wandb artifact cache getting full in /home/aviralku/.cache/wandb/artifacts
export WANDB_DIR=/project/flame/aviralku/wandb_logs
export WANDB_CACHE_DIR=/project/flame/aviralku/wandb_cache
export WANDB_ARTIFACT_DIR=/project/flame/aviralku/wandb_artifacts
export WANDB_DATA_DIR=/project/flame/aviralku/wandb_data
export HF_HOME=/project/flame/aviralku/hf_cache

timestamp="$(date +'%Y%m%d-%H%M%S')"

cd "${SCRIPT_DIR}"
python -m pipelinerl.launch \
  --config-name=genvf_v8_proof_unverified \
  output_dir="/project/flame/aviralku/results/genvf-v8-proof-unverified-20260408-170126" \
  llm_grader.name="openai/gpt-oss-20b" \
  world.actor_fraction=4 \
  world.finetune_fraction=4 \
  finetune.rl.entropy_bonus=0.0 \
  max_lag=1024

  # output_dir="/project/flame/aviralku/results/genvf-v8-proof-unverified-${timestamp}" \