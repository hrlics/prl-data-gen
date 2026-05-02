export OPENAI_BASE_URL=http://10.16.15.201:8000/v1
export OPENAI_API_KEY=grader

source /project/flame/aviralku/envs/prl/bin/activate

rm -rf /home/aviralku/.cache/wandb/artifacts # to avoid wandb artifact cache getting full

# to avoid wandb artifact cache getting full in /home/aviralku/.cache/wandb/artifacts
export WANDB_DIR=/project/flame/aviralku/wandb_logs
export WANDB_CACHE_DIR=/project/flame/aviralku/wandb_cache
export WANDB_ARTIFACT_DIR=/project/flame/aviralku/wandb_artifacts
export WANDB_DATA_DIR=/project/flame/aviralku/wandb_data

export HF_TOKEN=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
hf auth login --token $HF_TOKEN


# timestamp=$(date +'%Y%m%d-%H%M%S'); python -m pipelinerl.launch --config-name=genvf_v3_summary_hard_stage_2nd_wo_oracle.yaml output_dir="/project/flame/aviralku/results/genvg_v3_summary-hard_stage_2nd_purely_wo_oracle-${timestamp}"

# cotinue old run
python -m pipelinerl.launch \
  --config-name=genvf_v3_summary_hard_stage_2nd_wo_oracle \
  output_dir="/project/flame/aviralku/results/genvg_v3_summary-hard_stage_2nd_purely_wo_oracle-20260220-231334" \
  finetune.max_train_steps=336 \
  force_restart=false