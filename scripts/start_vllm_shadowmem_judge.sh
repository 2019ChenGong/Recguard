#!/usr/bin/env bash
# Judge server for the ShadowMem baseline (QWEN_DEFENSE=mage).
# Serves the checkpoint released by the ShadowMem authors (HuntingQuasar/pi-model) with plain
# chat completions, bfloat16 and a 32K context. No --generation-config override is given, so
# sampling follows the checkpoint's generation_config.json, as in the authors' runs.
# Clients read the endpoint from configs/shadowmem_judge.env.
#
#   CUDA_VISIBLE_DEVICES=1 scripts/start_vllm_shadowmem_judge.sh
set -euo pipefail

VLLM_PYTHON="${VLLM_PYTHON:-python}"          # python of the environment that has vllm installed
MODEL="${MAGE_JUDGE_MODEL:-HuntingQuasar/pi-model}"
REVISION="${MAGE_JUDGE_REVISION:-371487e37ff7b28a1e6a3dc008182f37615410e9}"
PORT="${PORT:-8010}"
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
    --served-model-name shadowmem-pi-judge \
    --download-dir "$HF_HUB_CACHE" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-32}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
    "$@"
