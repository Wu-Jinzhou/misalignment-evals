#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.cache/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"
export MPLCONFIGDIR

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/misalignment_eval_suite}"
TOKENIZER_ID="${TOKENIZER_ID:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-${OPENAI_MODEL:-${ANTHROPIC_MODEL:-gpt-5.4}}}"
JUDGE_MAX_WORKERS="${JUDGE_MAX_WORKERS:-8}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-1024}"
REWARD_HACKING_PROMPT_VARIANT="${REWARD_HACKING_PROMPT_VARIANT:-no_hints}"
REWARD_HACKING_REPEATS="${REWARD_HACKING_REPEATS:-5}"
BETLEY_REPEATS="${BETLEY_REPEATS:-2}"
GOALS_SAMPLES="${GOALS_SAMPLES:-20}"
ALIGNMENT_QUESTIONS_LIMIT="${ALIGNMENT_QUESTIONS_LIMIT:-200}"
MONITOR_SAMPLES="${MONITOR_SAMPLES:-20}"
EXFIL_SAMPLES="${EXFIL_SAMPLES:-20}"
FRAME_SAMPLES="${FRAME_SAMPLES:-20}"
EVAL_SYSTEM_PROMPT_PREFIX="${EVAL_SYSTEM_PROMPT_PREFIX:-}"
MODEL_SPECS="${MODEL_SPECS:-}"
CHECKPOINT_ROOTS="${CHECKPOINT_ROOTS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cmd=(
  python
  "${SCRIPT_DIR}/run_local_misalignment_eval_suite.py"
  --output-dir "${OUTPUT_DIR}"
  --batch-size "${BATCH_SIZE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --judge-model "${JUDGE_MODEL}"
  --judge-max-workers "${JUDGE_MAX_WORKERS}"
  --judge-max-tokens "${JUDGE_MAX_TOKENS}"
  --reward-hacking-prompt-variant "${REWARD_HACKING_PROMPT_VARIANT}"
  --reward-hacking-repeats "${REWARD_HACKING_REPEATS}"
  --betley-repeats "${BETLEY_REPEATS}"
  --goals-samples "${GOALS_SAMPLES}"
  --alignment-questions-limit "${ALIGNMENT_QUESTIONS_LIMIT}"
  --monitor-samples "${MONITOR_SAMPLES}"
  --exfil-samples "${EXFIL_SAMPLES}"
  --frame-samples "${FRAME_SAMPLES}"
  --eval-system-prompt-prefix "${EVAL_SYSTEM_PROMPT_PREFIX}"
)

if [[ -n "${TOKENIZER_ID}" ]]; then
  cmd+=(--tokenizer-id "${TOKENIZER_ID}")
fi

if [[ "${ENABLE_THINKING}" == "1" ]]; then
  cmd+=(--enable-thinking)
else
  cmd+=(--no-enable-thinking)
fi

for spec in ${MODEL_SPECS}; do
  cmd+=(--model "${spec}")
done

for spec in ${CHECKPOINT_ROOTS}; do
  cmd+=(--checkpoint-root "${spec}")
done

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_ARGS} )
  cmd+=("${extra[@]}")
fi

"${cmd[@]}"
