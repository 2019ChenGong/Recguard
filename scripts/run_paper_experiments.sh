#!/usr/bin/env bash
# Run the experiments of the paper, one group at a time.
#
#   scripts/run_paper_experiments.sh <group> [<group> ...]
#   DRY_RUN=1 scripts/run_paper_experiments.sh existing      # print the commands only
#   FRAMEWORKS="local" scripts/run_paper_experiments.sh ours # skip Letta and OpenAI-SM
#
# Groups
#   smoke       one user, two categories, 20 everyday messages
#   existing    no defense and the seven existing defenses under SFP and TPB
#   shadowmem   the ShadowMem baseline (needs the judge server)
#   label       source label stored in memory, source filter before memory, forwarded messages
#   ours        MailRecGuard under SFP, TPB, FWD and on a clean inbox
#   ablation    MailRecGuard without one component; K = 1 and K = 3
#   labelerror  wrong source labels (5 %, 10 %, 20 %; three error patterns)
#   count       number of attack messages per category
#   size        number of everyday messages per user
#   dormancy    all attack messages arrive before the everyday messages
#
# FRAMEWORKS selects the framework sets: "local" (Explicit, Context, RAG, Mem0, MemoryBank,
# TencentDB), "letta", "openai_sandbox". Default: all three. Every command is one call of
# scripts/run_grid.sh; the commands are independent and can be run in parallel on several
# model servers (set QWEN_ENV_FILE per command).
set -uo pipefail
cd "$(dirname "$0")/.."

LOCAL="explicit context rag mem0 memorybank tencentdb"
NAT="baseline_no_email poisoned_neutral_daily_inbox"
CLEAN="baseline_no_email clean_preference_neutral_daily_inbox"
OURS="typed_ledger+ownpart+conflict"
VARIANTS="none,amemguard,smsr,action_gate,strict_guard"   # one ingestion, one query pass per variant
SFP="spoofed_first_party"; TPB="legit_marketing"; FWD="forwarded_ad"
FRAMEWORKS="${FRAMEWORKS:-local letta openai_sandbox}"

want() { [[ " $FRAMEWORKS " == *" $1 "* ]]; }

run() {   # run <tag> KEY=VALUE ...
  local tag="$1"; shift
  if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "TAG=$tag $(printf '%q ' "$@")scripts/run_grid.sh"
  else
    env TAG="$tag" "$@" bash scripts/run_grid.sh
  fi
}

# A defense that leaves the whole inbox to the framework. Explicit, Context and RAG store the
# agent's own notes, so they run in one process and share the writer's calls.
full() {  # full <tag> <attack> <defense> <conditions> [KEY=VALUE ...]
  local tag="$1" style="$2" defense="$3" cond="$4"; shift 4
  local common=(STYLES="$style" DEFENSES="$defense" CONDITIONS="$cond" "$@")
  if want local; then
    run "${tag}_ecr" BACKENDS="explicit context rag" QWEN_SHARE_WRITER_NOTES=1 QWEN_MAX_CONCURRENT_RECS=8 "${common[@]}"
    for b in tencentdb memorybank mem0; do
      run "${tag}_${b}" BACKENDS="$b" QWEN_MAX_CONCURRENT_RECS=8 "${common[@]}"
    done
  fi
  if want letta; then run "${tag}_letta" BACKENDS=letta QWEN_MAX_CONCURRENT_RECS=6 "${common[@]}"; fi
  if want openai_sandbox; then run "${tag}_openai_sandbox" BACKENDS=openai_sandbox QWEN_MAX_CONCURRENT_RECS=6 "${common[@]}"; fi
}

# A defense that passes only a few records to the framework: all local frameworks in one process.
light() { # light <tag> <attack> <defense> <conditions> [KEY=VALUE ...]
  local tag="$1" style="$2" defense="$3" cond="$4"; shift 4
  local common=(STYLES="$style" DEFENSES="$defense" CONDITIONS="$cond" "$@")
  if want local; then run "${tag}_local" BACKENDS="$LOCAL" QWEN_SHARE_WRITER_NOTES=1 QWEN_MAX_CONCURRENT_RECS=8 "${common[@]}"; fi
  if want letta; then run "${tag}_letta" BACKENDS=letta QWEN_MAX_CONCURRENT_RECS=6 "${common[@]}"; fi
  if want openai_sandbox; then run "${tag}_openai_sandbox" BACKENDS=openai_sandbox QWEN_MAX_CONCURRENT_RECS=6 "${common[@]}"; fi
}

[ $# -ge 1 ] || { sed -n 2,24p "$0"; exit 1; }
for group in "$@"; do
  case "$group" in
    smoke)
      for d in none "$OURS"; do
        run "smoke" BACKENDS="explicit rag" STYLES="$SFP" DEFENSES="$d" CONDITIONS="$NAT" \
            PERSONAS=p01 CATS="coffee_maker running_shoes" QWEN_DAILY_BENIGN_COUNT=20
      done
      ;;
    existing)
      for s in "$SFP" "$TPB"; do
        # no defense + A-MemGuard, SMSR, PPMF, counterfactual guard (they act at query time)
        full "none_${s}" "$s" none "$NAT" QWEN_QUERY_VARIANTS="$VARIANTS"
        # Memory Sanitization, StateGuard, Trojan Hippo (they act before or at the memory write)
        for d in content_classifier memory_stateguard limit_memory_length; do
          full "${d}_${s}" "$s" "$d" "$NAT"
        done
      done
      full "none_clean" "$SFP" none "$CLEAN" QWEN_QUERY_VARIANTS="$VARIANTS"
      ;;
    shadowmem)
      source "${MAGE_ENV_FILE:-configs/shadowmem_judge.env}"
      for s in "$SFP" "$TPB"; do full "mage_${s}" "$s" mage "$NAT"; done
      ;;
    label)
      for s in "$SFP" "$TPB"; do
        full  "stamp_${s}"  "$s" channel_stamp "$NAT"     # label stored in memory
        light "filter_${s}" "$s" channel_taint "$NAT"     # filter before memory
      done
      light "filter_${FWD}" "$FWD" channel_taint "$NAT"
      full  "none_${FWD}"   "$FWD" none "$NAT"
      light "filter_clean"  "$SFP" channel_taint "$CLEAN"
      ;;
    ours)
      for s in "$SFP" "$TPB" "$FWD"; do light "ours_${s}" "$s" "$OURS" "$NAT"; done
      light "ours_clean" "$SFP" "$OURS" "$CLEAN"
      ;;
    ablation)
      for s in "$SFP" "$TPB" "$FWD"; do
        light "abl_nosource_${s}"   "$s" "$OURS+noauth"          "$NAT"   # without the source check
        light "abl_noown_${s}"      "$s" "typed_ledger+conflict" "$NAT"   # without the own-text check
        light "abl_nostance_${s}"   "$s" "$OURS+nostance"        "$NAT"   # without the stance check
        light "abl_noconflict_${s}" "$s" "typed_ledger+ownpart"  "$NAT"   # without conflict confirmation
        light "abl_k1_${s}"         "$s" "$OURS+k1"              "$NAT"
        light "abl_k3_${s}"         "$s" "$OURS+k3"              "$NAT"
      done
      ;;
    labelerror)
      for p in 0.05 0.1 0.2; do
        for seed in 1 2 3; do
          for d in channel_taint "typed_ledger+ownpart" "$OURS" "typed_ledger+ownpart+conflictna"; do
            for s in "$SFP" "$TPB"; do
              # false accepts and false rejects at rate p
              FRAMEWORKS=local light "err${p}_r${seed}_${s}" "$s" "$d" "$NAT" \
                  QWEN_CHANNEL_NOISE="$p" QWEN_CHANNEL_NOISE_SEED="$seed"
              # false accepts only
              FRAMEWORKS=local light "fa${p}_r${seed}_${s}" "$s" "$d" "poisoned_neutral_daily_inbox" \
                  QWEN_CHANNEL_NOISE="$p" QWEN_CHANNEL_NOISE_FR=0 QWEN_CHANNEL_NOISE_SEED="$seed"
            done
          done
        done
      done
      ;;
    count)
      for n in 1 5 10; do
        FRAMEWORKS=local light "ours_sfp${n}" "$SFP" "$OURS" "$NAT" QWEN_SPOOF_COUNT="$n"
        FRAMEWORKS=local full  "none_sfp${n}" "$SFP" none    "$NAT" QWEN_SPOOF_COUNT="$n"
      done
      for n in 1 3 10; do
        FRAMEWORKS=local light "ours_tpb${n}" "$TPB" "$OURS" "$NAT" QWEN_TPB_COUNT="$n"
        FRAMEWORKS=local full  "none_tpb${n}" "$TPB" none    "$NAT" QWEN_TPB_COUNT="$n"
      done
      ;;
    size)
      for n in 0 100 1000; do
        for s in "$SFP" "$TPB"; do
          FRAMEWORKS=local light "ours_d${n}_${s}" "$s" "$OURS" "$NAT" QWEN_DAILY_BENIGN_COUNT="$n"
          FRAMEWORKS=local full  "none_d${n}_${s}" "$s" none    "$NAT" QWEN_DAILY_BENIGN_COUNT="$n"
        done
      done
      ;;
    dormancy)
      for n in 100 300 1000; do
        for s in "$SFP" "$TPB"; do
          FRAMEWORKS=local light "ours_dorm${n}_${s}" "$s" "$OURS" "$NAT" \
              QWEN_DAILY_BENIGN_COUNT="$n" QWEN_DAILY_ORDER=attack_first
          FRAMEWORKS=local full  "none_dorm${n}_${s}" "$s" none    "$NAT" \
              QWEN_DAILY_BENIGN_COUNT="$n" QWEN_DAILY_ORDER=attack_first
        done
      done
      ;;
    *)
      echo "unknown group: $group"; sed -n 2,24p "$0"; exit 1 ;;
  esac
done
