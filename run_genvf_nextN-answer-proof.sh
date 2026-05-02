#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "/project/flame/aviralku/envs/prl/bin/activate"

# === local grader config ===
export OPENAI_BASE_URL=http://10.16.0.81:8000/v1
export OPENAI_API_KEY=grader

# to avoid wandb artifact cache getting full in /home/aviralku/.cache/wandb/artifacts
export WANDB_DIR=/project/flame/aviralku/wandb_logs
export WANDB_CACHE_DIR=/project/flame/aviralku/wandb_cache
export HF_HOME=/tmp/hf_cache # if use this, must export HF_token, otherwise it can't find HF token under ~/.cache/huggingface/token
export HF_TOKEN="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"

timestamp="$(date +'%Y%m%d-%H%M%S')"

cd "${SCRIPT_DIR}"

# resart from 300 steps checkpoint
python -m pipelinerl.launch \
  --config-name=genvf_v9_1_answer_proof_nextN_300_start \
  output_dir="/project/flame/aviralku/results/genvf-v9_1-answer-proof-nextN-300-start-${timestamp}" \
  llm_grader.name="openai/gpt-oss-20b" \
  world.actor_fraction=4 \
  world.finetune_fraction=4 \
  finetune.rl.entropy_bonus=0.0 \
  finetune.rl.epsilon=1 \
  max_lag=1024


# python -m pipelinerl.launch \
#   --config-name=genvf_v9_1_answer_proof_nextN \
#   output_dir="/project/flame/aviralku/results/genvf-v9_1-answer-proof-nextN-20260413-221505" \
#   llm_grader.name="openai/gpt-oss-20b" \
#   world.actor_fraction=4 \
#   world.finetune_fraction=4 \
#   finetune.rl.entropy_bonus=0.0 \
#   max_lag=1024


#output_dir="/project/flame/aviralku/results/genvf-v9_1-answer-proof-nextN-${timestamp}" \



# === remote grader config ===
# export OPENAI_BASE_URL="https://mtllv6vkucczkopr.us-east-2.aws.endpoints.huggingface.cloud/v1"
# export OPENAI_API_KEY="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"