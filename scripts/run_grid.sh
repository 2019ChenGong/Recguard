#!/usr/bin/env bash
# Run one grid: every (attack, defense) pair on the selected memory frameworks.
# All settings are environment variables; the defaults give the paper's main setting
# (natural inbox: 300 everyday messages, one genuine note per category, 3 SFP or 5 TPB
# attack messages per category, 3 users x 10 queried categories = 30 queries per cell).
#
#   STYLES="spoofed_first_party" DEFENSES="none" BACKENDS="explicit rag" scripts/run_grid.sh
#
#   STYLES      attack scenario(s): spoofed_first_party (SFP), legit_marketing (TPB), forwarded_ad (FWD)
#   DEFENSES    defense(s), see README.md ("Names in the paper and in the code")
#   BACKENDS    memory framework(s): explicit context rag mem0 memorybank tencentdb letta openai_sandbox
#   CONDITIONS  inbox condition(s): baseline_no_email, clean_preference_neutral_daily_inbox,
#               poisoned_neutral_daily_inbox
#   PERSONAS / CATS   users and queried product categories
#   TAG         prefix of the log files in logs/
set -uo pipefail
cd "$(dirname "$0")/.."
source "${QWEN_ENV_FILE:-configs/vllm.env}"      # endpoint of the vLLM server
export PYTHONNOUSERSITE=1
export QWEN_CATALOG_V2="${QWEN_CATALOG_V2:-1}"
export QWEN_RECOMMENDER="${QWEN_RECOMMENDER:-raw}"
export QWEN_HM_EMAILS_PER_CATEGORY="${QWEN_HM_EMAILS_PER_CATEGORY:-1}"   # genuine notes per category
export QWEN_SPOOF_COUNT="${QWEN_SPOOF_COUNT:-3}"                         # SFP messages per category
export QWEN_TPB_COUNT="${QWEN_TPB_COUNT:-5}"                             # TPB messages per category
export QWEN_DAILY_BENIGN_COUNT="${QWEN_DAILY_BENIGN_COUNT:-300}"         # everyday messages per user
# OpenAI-SM: 10 messages per sandbox batch keeps each rollout transcript small. The
# consolidation prompt of the SDK alone is about 12k tokens, so the server needs a 32K context.
export OPENAI_SANDBOX_MEMORY_BATCH_SIZE="${OPENAI_SANDBOX_MEMORY_BATCH_SIZE:-10}"
# Keep the BLAS / tokenizer thread pools small; several drivers can then share one machine.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" TOKENIZERS_PARALLELISM=false
PERSONAS="${PERSONAS:-p01 p02 p03}"
CATS="${CATS:-electric_toothbrush wireless_earbuds running_shoes coffee_maker air_purifier office_chair robot_vacuum laptop_backpack blender smart_watch}"
BACKENDS="${BACKENDS:-explicit context rag mem0 memorybank tencentdb}"
STYLES="${STYLES:-spoofed_first_party legit_marketing}"
DEFENSES="${DEFENSES:-none}"
CONDITIONS="${CONDITIONS:-baseline_no_email poisoned_neutral_daily_inbox}"
TAG="${TAG:-grid}"
PYTHON="${PYTHON:-python}"
mkdir -p logs
for style in $STYLES; do
  for defense in $DEFENSES; do
    log="logs/${TAG}_${style}_${defense}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== style=$style defense=$defense backends=$BACKENDS $(date) -> $log"
    QWEN_EMAIL_STYLE="$style" QWEN_DEFENSE="$defense" \
      "$PYTHON" scripts/exp_backends_qwen.py --backends $BACKENDS \
        --conditions $CONDITIONS \
        --personas $PERSONAS --categories $CATS > "$log" 2>&1
    echo "rc=$? $(grep -cE 'Traceback' "$log") tracebacks; summary: $(grep -oE 'backends_grid_summary_[0-9_]+\.json' "$log" | tail -1)"
  done
done
echo "ALL DONE $(date)"
