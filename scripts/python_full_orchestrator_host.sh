#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/master-cluster-HSE}"

BEST_4B_CL="cl_semantic_distribution"

BASE4="/workspace/models/qwen3-4b-instruct-2507"
BASE8="/workspace/models/qwen3-8b"

SLUG4="qwen3-4b-instruct-2507"
SLUG8="qwen3-8b"

GRPO_JOB="jobs/rl_v100_vllm085_grpo_server_2gpu_normal.sbatch"
if [[ ! -f "$GRPO_JOB" ]]; then
  GRPO_JOB="jobs/rl_v100_vllm085_grpo_server_2gpu.sbatch"
fi
if [[ ! -f "$GRPO_JOB" ]]; then
  echo "No 2GPU GRPO job found" >&2
  exit 1
fi

mkdir -p logs/slurm outputs/pipeline_python

write_lines() {
  local path="$1"
  shift
  mkdir -p "$(dirname "$path")"
  : > "$path"
  for x in "$@"; do
    echo "$x" >> "$path"
  done
}

submit_train_matrix() {
  local phase="$1"
  local selected_model="$2"
  local throttle="$3"
  local dependency="${4:-}"
  shift 4 || true
  local runs=("$@")

  local matrix="outputs/pipeline_python/${phase}/train_runs.txt"
  write_lines "$matrix" "${runs[@]}"

  local args=()
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="$dependency")
  fi

  sbatch --parsable \
    ${args[@]+"${args[@]}"} \
    --array="1-${#runs[@]}%${throttle}" \
    --export=ALL,MATRIX="$matrix",SELECTED_MODEL="$selected_model",CONFIG=configs/train/lora_pythoncodes_cl.yaml \
    jobs/python_train_matrix_rocky.sbatch
}

submit_eval_matrix() {
  local phase="$1"
  local selected_model="$2"
  local parts_name="$3"
  local out_name="$4"
  local throttle="$5"
  local dependency="${6:-}"
  shift 6 || true
  local exps=("$@")

  local matrix="outputs/pipeline_python/${phase}/${parts_name}.txt"
  local parts_root="outputs/pipeline_python/${phase}/${parts_name}"
  local out_dir="outputs/pipeline_python/${phase}/${out_name}"

  write_lines "$matrix" "${exps[@]}"

  local args=()
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="$dependency")
  fi

  local jid_eval
  jid_eval=$(sbatch --parsable \
    ${args[@]+"${args[@]}"} \
    --array="1-${#exps[@]}%${throttle}" \
    --export=ALL,MATRIX="$matrix",PARTS_ROOT="$parts_root",SELECTED_MODEL="$selected_model",EVAL_BENCHMARKS=mbpp,humaneval \
    jobs/python_eval_matrix_rocky.sbatch)

  local jid_merge
  jid_merge=$(sbatch --parsable \
    --dependency=afterok:"$jid_eval" \
    --export=ALL,PARTS_ROOT="$parts_root",OUT_DIR="$out_dir" \
    jobs/python_merge_eval_rocky.sbatch)

  echo "$jid_eval $jid_merge"
}

submit_grpo() {
  local model="$1"
  local slug="$2"
  local run_name="$3"
  local init_adapter="${4:-}"
  local dependency="${5:-}"
  local gpu_util="${6:-0.80}"

  local args=()
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="$dependency")
  fi

  local export_env="ALL,MODEL=${model},GRPO_RUN_NAME=${run_name},GRPO_ADAPTER_OUTPUT_DIR=/workspace/models/${slug}-sft-${run_name},SERVER_GPU_UTIL=${gpu_util},SERVER_MAX_MODEL_LEN=2048,SERVER_DTYPE=float16,SERVER_ENFORCE_EAGER=true"
  if [[ -n "$init_adapter" ]]; then
    export_env="${export_env},RL_INIT_ADAPTER=${init_adapter}"
  fi

  sbatch --parsable \
    ${args[@]+"${args[@]}"} \
    --export="$export_env" \
    "$GRPO_JOB"
}

phase4_submit() {
  echo "===== CLEAN 4B TARGETS ====="

  rm -rf \
    "models/${SLUG4}-sft-sft_pythoncodes" \
    "models/${SLUG4}-sft-${BEST_4B_CL}" \
    "models/${SLUG4}-sft-grpo_full_over_4b_base" \
    "models/${SLUG4}-sft-grpo_full_over_4b_sft_pythoncodes" \
    "models/${SLUG4}-sft-grpo_full_over_4b_${BEST_4B_CL}" \
    outputs/pipeline_python/4b

  echo "===== SUBMIT 4B TRAIN: SFT + BEST CL ====="

  local jid_train
  jid_train=$(submit_train_matrix \
    4b 4b-instruct 2 "" \
    sft_pythoncodes "$BEST_4B_CL")

  echo "jid_train_4b=$jid_train"

  echo "===== SUBMIT 4B PRE-RL EVAL ====="

  read -r jid_eval_pre jid_merge_pre < <(submit_eval_matrix \
    4b 4b-instruct pre_rl_eval_jobs pre_rl_eval 3 afterok:"$jid_train" \
    base sft_pythoncodes "$BEST_4B_CL")

  echo "jid_eval_pre_4b=$jid_eval_pre"
  echo "jid_merge_pre_4b=$jid_merge_pre"

  echo "===== SUBMIT 4B GRPO ====="

  local jid_rl_base jid_rl_sft jid_rl_cl

  jid_rl_base=$(submit_grpo \
    "$BASE4" "$SLUG4" grpo_full_over_4b_base "" afterok:"$jid_merge_pre" 0.80)

  jid_rl_sft=$(submit_grpo \
    "$BASE4" "$SLUG4" grpo_full_over_4b_sft_pythoncodes \
    "/workspace/models/${SLUG4}-sft-sft_pythoncodes" \
    afterok:"$jid_merge_pre" 0.80)

  # RL jobs must use at most 4 GPUs total: base + sft run together, CL after both.
  jid_rl_cl=$(submit_grpo \
    "$BASE4" "$SLUG4" "grpo_full_over_4b_${BEST_4B_CL}" \
    "/workspace/models/${SLUG4}-sft-${BEST_4B_CL}" \
    afterok:"${jid_rl_base}:${jid_rl_sft}" 0.80)

  echo "jid_rl_base_4b=$jid_rl_base"
  echo "jid_rl_sft_4b=$jid_rl_sft"
  echo "jid_rl_cl_4b=$jid_rl_cl"

  echo "===== SUBMIT 4B FINAL EVAL ====="

  read -r jid_eval_final jid_merge_final < <(submit_eval_matrix \
    4b 4b-instruct final_eval_jobs final_eval 4 afterok:"$jid_rl_cl" \
    base \
    sft_pythoncodes \
    "$BEST_4B_CL" \
    grpo_full_over_4b_base \
    grpo_full_over_4b_sft_pythoncodes \
    "grpo_full_over_4b_${BEST_4B_CL}")

  echo "jid_eval_final_4b=$jid_eval_final"
  echo "jid_merge_final_4b=$jid_merge_final"
  echo "final_4b_table=outputs/pipeline_python/4b/final_eval/lora_eval_comparison.csv"
}

phase4_clean() {
  echo "===== CLEAN 4B ADAPTERS, KEEP TABLES ====="

  rm -rf \
    "models/${SLUG4}-sft-sft_pythoncodes" \
    "models/${SLUG4}-sft-${BEST_4B_CL}" \
    "models/${SLUG4}-sft-grpo_full_over_4b_base" \
    "models/${SLUG4}-sft-grpo_full_over_4b_sft_pythoncodes" \
    "models/${SLUG4}-sft-grpo_full_over_4b_${BEST_4B_CL}"

  echo "kept outputs/pipeline_python/4b"
}

phase8_train_eval_submit() {
  echo "===== CLEAN 8B TRAIN/EVAL TARGETS ====="

  rm -rf outputs/pipeline_python/8b

  local runs=(
    sft_pythoncodes
    cl_length_staged
    cl_perplexity_staged
    cl_lexical_staged
    cl_semantic_staged
    cl_llm_judge_staged
    cl_length_cumulative
    cl_perplexity_cumulative
    cl_lexical_cumulative
    cl_semantic_cumulative
    cl_llm_judge_cumulative
    cl_length_distribution
    cl_perplexity_distribution
    cl_lexical_distribution
    cl_semantic_distribution
    cl_llm_judge_distribution
  )

  for r in "${runs[@]}"; do
    rm -rf "models/${SLUG8}-sft-${r}"
  done

  echo "===== SUBMIT 8B TRAIN: SFT + ALL 15 CL ====="

  local jid_train
  jid_train=$(submit_train_matrix 8b 8b 4 "" "${runs[@]}")
  echo "jid_train_8b=$jid_train"

  echo "===== SUBMIT 8B ALL-CL EVAL ====="

  read -r jid_eval jid_merge < <(submit_eval_matrix \
    8b 8b all_cl_eval_jobs all_cl_eval 4 afterok:"$jid_train" \
    base "${runs[@]}")

  echo "jid_eval_8b=$jid_eval"
  echo "jid_merge_8b=$jid_merge"
  echo "all_cl_table=outputs/pipeline_python/8b/all_cl_eval/lora_eval_comparison.csv"
}

phase8_pick_best() {
  echo "===== PICK BEST 8B CL ====="

  scripts/run_container_python_login.sh '
    python scripts/python_pick_best_cl.py \
      --csv outputs/pipeline_python/8b/all_cl_eval/lora_eval_comparison.csv \
      --out outputs/pipeline_python/8b/best_cl.txt
  '

  echo "best_cl=$(cat outputs/pipeline_python/8b/best_cl.txt)"
}

phase8_rl_submit() {
  echo "===== 8B RL SUBMIT ====="

  local best
  best="$(cat outputs/pipeline_python/8b/best_cl.txt)"
  if [[ -z "$best" || "$best" != cl_* ]]; then
    echo "bad best CL: '$best'" >&2
    exit 1
  fi

  echo "best_cl=$best"

  echo "===== DELETE UNNEEDED 8B CL ADAPTERS ====="

  local runs=(
    cl_length_staged
    cl_perplexity_staged
    cl_lexical_staged
    cl_semantic_staged
    cl_llm_judge_staged
    cl_length_cumulative
    cl_perplexity_cumulative
    cl_lexical_cumulative
    cl_semantic_cumulative
    cl_llm_judge_cumulative
    cl_length_distribution
    cl_perplexity_distribution
    cl_lexical_distribution
    cl_semantic_distribution
    cl_llm_judge_distribution
  )

  for r in "${runs[@]}"; do
    if [[ "$r" != "$best" ]]; then
      rm -rf "models/${SLUG8}-sft-${r}"
    fi
  done

  rm -rf \
    "models/${SLUG8}-sft-grpo_full_over_8b_base" \
    "models/${SLUG8}-sft-grpo_full_over_8b_sft_pythoncodes" \
    "models/${SLUG8}-sft-grpo_full_over_8b_${best}"

  echo "===== SUBMIT 8B GRPO ====="

  local jid_rl_base jid_rl_sft jid_rl_cl

  jid_rl_base=$(submit_grpo \
    "$BASE8" "$SLUG8" grpo_full_over_8b_base "" "" 0.75)

  jid_rl_sft=$(submit_grpo \
    "$BASE8" "$SLUG8" grpo_full_over_8b_sft_pythoncodes \
    "/workspace/models/${SLUG8}-sft-sft_pythoncodes" \
    "" 0.75)

  # RL jobs must use at most 4 GPUs total: base + sft run together, CL after both.
  jid_rl_cl=$(submit_grpo \
    "$BASE8" "$SLUG8" "grpo_full_over_8b_${best}" \
    "/workspace/models/${SLUG8}-sft-${best}" \
    afterok:"${jid_rl_base}:${jid_rl_sft}" 0.75)

  echo "jid_rl_base_8b=$jid_rl_base"
  echo "jid_rl_sft_8b=$jid_rl_sft"
  echo "jid_rl_cl_8b=$jid_rl_cl"

  echo "===== SUBMIT 8B FINAL EVAL ====="

  read -r jid_eval_final jid_merge_final < <(submit_eval_matrix \
    8b 8b final_eval_jobs final_eval 4 afterok:"$jid_rl_cl" \
    base \
    sft_pythoncodes \
    "$best" \
    grpo_full_over_8b_base \
    grpo_full_over_8b_sft_pythoncodes \
    "grpo_full_over_8b_${best}")

  echo "jid_eval_final_8b=$jid_eval_final"
  echo "jid_merge_final_8b=$jid_merge_final"
  echo "final_8b_table=outputs/pipeline_python/8b/final_eval/lora_eval_comparison.csv"
}

phase8_clean() {
  echo "===== CLEAN 8B ADAPTERS, KEEP TABLES ====="

  local best=""
  if [[ -f outputs/pipeline_python/8b/best_cl.txt ]]; then
    best="$(cat outputs/pipeline_python/8b/best_cl.txt)"
  fi

  rm -rf \
    "models/${SLUG8}-sft-sft_pythoncodes" \
    "models/${SLUG8}-sft-grpo_full_over_8b_base" \
    "models/${SLUG8}-sft-grpo_full_over_8b_sft_pythoncodes"

  if [[ -n "$best" ]]; then
    rm -rf \
      "models/${SLUG8}-sft-${best}" \
      "models/${SLUG8}-sft-grpo_full_over_8b_${best}"
  fi

  echo "kept outputs/pipeline_python/8b"
}

case "${1:-}" in
  phase4_submit) phase4_submit ;;
  phase4_clean) phase4_clean ;;
  phase8_train_eval_submit) phase8_train_eval_submit ;;
  phase8_pick_best) phase8_pick_best ;;
  phase8_rl_submit) phase8_rl_submit ;;
  phase8_clean) phase8_clean ;;
  *)
    echo "Usage: $0 {phase4_submit|phase4_clean|phase8_train_eval_submit|phase8_pick_best|phase8_rl_submit|phase8_clean}" >&2
    exit 2
    ;;
esac
