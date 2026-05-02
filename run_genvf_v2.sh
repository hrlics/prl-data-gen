#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "/project/flame/aviralku/envs/prl/bin/activate"

# # === local grader config ===
# export OPENAI_BASE_URL=http://10.16.0.81:8000/v1
# export OPENAI_API_KEY=grader

# remote grader config
export OPENAI_BASE_URL="https://mtllv6vkucczkopr.us-east-2.aws.endpoints.huggingface.cloud/v1"
export OPENAI_API_KEY="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"

# === wandb config ===
export WANDB_DIR=/tmp/wandb_logs
export WANDB_CACHE_DIR=/tmp/wandb_cache
export HF_HOME=/tmp/hf_cache
export HF_TOKEN="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"

timestamp="$(date +'%Y%m%d-%H%M%S')"

# 训练写本地 SSD,job 结束(正常或被杀)前 rsync 回 flame
RUN_NAME="genvf-v8-tcs-4B-v2-${timestamp}"
LOCAL_OUT="/tmp/aviralku/results/${RUN_NAME}"
FINAL_OUT="/project/flame/aviralku/results/${RUN_NAME}"
mkdir -p "${LOCAL_OUT}"
sync_back() {
  echo "[$(date)] syncing ${LOCAL_OUT} -> ${FINAL_OUT}"
  mkdir -p "${FINAL_OUT}"
  rsync -a --info=progress2 "${LOCAL_OUT}/" "${FINAL_OUT}/" || true
}
trap sync_back EXIT INT TERM

cd "${SCRIPT_DIR}"
python -m pipelinerl.launch \
  --config-name=genvf_v8_tcs_v2 \
  output_dir="${LOCAL_OUT}" \
  llm_grader.name="openai/gpt-oss-20b" \
  world.actor_fraction=4 \
  world.finetune_fraction=4 \
  finetune.rl.entropy_bonus=0.0 \
  finetune.rl.epsilon=1 \
  max_lag=1024


# output_dir="/project/flame/aviralku/results/genvf-v8-${timestamp}" \