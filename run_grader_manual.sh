# compilation does not play nice with Ray
export TORCH_COMPILE_DISABLE=1

set -x -e

cd /home/aviralku/haoranl4/pipeline-rl
source /project/flame/aviralku/envs/grader/bin/activate
export VLLM_API_KEY=grader

HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
RAY_PORT=6379
echo "OPENAI_BASE_URL=http://${HEAD_NODE_IP}:8000/v1"

# clean old ray processes and logs
ray stop --force || true

# start Ray head node
ray start --head \
  --node-ip-address="$HEAD_NODE_IP" \
  --port=$RAY_PORT \
  --num-cpus=88 \
  --num-gpus=8

# launch vllm（data parallel = 8, TP = 1）
# NOTE: need to change max-model-len according to prefix length and output length
# here the prefix leng<=25k and the output is bullet list, so we set max-model-len to 65k to be safe
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve openai/gpt-oss-20b \
  --host 0.0.0.0 \
  --port 8000 \
  --distributed-executor-backend ray \
  --data-parallel-backend ray \
  --data-parallel-address "$HEAD_NODE_IP" \
  --data-parallel-size 8 \
  --data-parallel-size-local 8 \
  --tensor-parallel-size 1 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 24576 \
  --max-num-seqs 64 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9