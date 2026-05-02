#!/bin/bash
# Prereqs (one-time): python tests/subsample.py    # creates haoranli-ml/genvf-data-generator-100prefix-v1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "/project/flame/aviralku/envs/prl/bin/activate"

# === Judge endpoint ===
# (B) Remote HF Inference Endpoint (matches existing run_genvf.sh).
export OPENAI_BASE_URL="https://mtllv6vkucczkopr.us-east-2.aws.endpoints.huggingface.cloud/v1"
export OPENAI_API_KEY="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"

# === wandb / hf cache ===
export WANDB_DIR=/tmp/wandb_logs
export WANDB_CACHE_DIR=/tmp/wandb_cache
export HF_HOME=/tmp/hf_cache
export HF_TOKEN="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"

timestamp="$(date +'%Y%m%d-%H%M%S')"
RUN_NAME="genvf-v10-data-generator-4B-${timestamp}"
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
  --config-name=genvf_v10_data_generator \
  output_dir="${LOCAL_OUT}" \
  llm_grader.name="openai/gpt-oss-20b" \
  world.actor_fraction=4 \
  world.finetune_fraction=4 \
  finetune.rl.entropy_bonus=0.0 \
  finetune.rl.epsilon=1 \
  max_lag=1024 \
  "$@"
