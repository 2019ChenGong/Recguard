#!/usr/bin/env bash
# Serve Qwen3.8-27B (FP8 checkpoint, pinned revision) with vLLM for all experiments.
#
#   one 48 GB GPU :  CUDA_VISIBLE_DEVICES=0 scripts/start_vllm_qwen.sh
#   two 24 GB GPUs:  CUDA_VISIBLE_DEVICES=0,1 VLLM_TENSOR_PARALLEL_SIZE=2 \
#                    VLLM_GPU_MEMORY_UTILIZATION=0.92 scripts/start_vllm_qwen.sh
#
# Clients read the endpoint from configs/vllm.env (see configs/vllm.env.example).
# The 32K context is required by the OpenAI-SM and Letta frameworks.
set -euo pipefail

VLLM_PYTHON="${VLLM_PYTHON:-python}"          # python of the environment that has vllm installed
MODEL="${VLLM_MODEL:-Qwen/Qwen3.8-27B-FP8}"
REVISION="${VLLM_MODEL_REVISION:-017b9c7af6b5689d5dd426a76e0bc077eb5ca20a}"
PORT="${PORT:-8000}"
API_KEY="${VLLM_API_KEY:-token-vllm}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"
export PYTHONNOUSERSITE=1
mkdir -p "$HF_HUB_CACHE"

printf 'Model: %s\nRevision: %s\nGPUs: %s\nEndpoint: http://%s:%s/v1\n' \
    "$MODEL" "$REVISION" "$CUDA_VISIBLE_DEVICES" "$(hostname)" "$PORT"

exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --revision "$REVISION" \
    --tokenizer-revision "$REVISION" \
    --served-model-name "$MODEL" \
    --download-dir "$HF_HUB_CACHE" \
    --safetensors-load-strategy eager \
    --host 0.0.0.0 \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE:-1}" \
    --dtype bfloat16 \
    --language-model-only \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-16}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.88}" \
    --generation-config vllm \
    --seed 0 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking":false,"preserve_thinking":false}' \
    "$@"
