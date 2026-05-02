export OPENAI_BASE_URL=http://10.16.0.77:8000/v1
export OPENAI_API_KEY=grader

source /project/flame/aviralku/envs/prl/bin/activate

rm -rf /home/aviralku/.cache/wandb/artifacts # to avoid wandb artifact cache getting full

# to avoid wandb artifact cache getting full in /home/aviralku/.cache/wandb/artifacts
export WANDB_DIR=/project/flame/aviralku/wandb_logs
export WANDB_CACHE_DIR=/project/flame/aviralku/wandb_cache
export WANDB_ARTIFACT_DIR=/project/flame/aviralku/wandb_artifacts
export WANDB_DATA_DIR=/project/flame/aviralku/wandb_data


timestamp=$(date +'%Y%m%d-%H%M%S'); python -m pipelinerl.launch --config-name=genvf_v4 output_dir="/project/flame/aviralku/results/genvf_v4-${timestamp}"