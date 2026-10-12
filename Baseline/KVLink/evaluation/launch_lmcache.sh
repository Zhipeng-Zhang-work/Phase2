#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/chzhan.chzhipen/LMCache}"
MODEL="${MODEL:-$REPO_ROOT/Baseline/checkpoints/kvlink-qwen32b}"
DATASET="${DATASET:-$REPO_ROOT/benchmarks/rag/inputs/longbench_inputs/2wikimqa_longbench.json}"
PROMPT_BUILD_METHOD="${PROMPT_BUILD_METHOD:-QA}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/Baseline/KVLink/evaluation/cache}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/Baseline/KVLink/evaluation/results}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
QPS="${QPS:-0.25}"

cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"
mkdir -p "$CACHE_DIR" "$RESULT_DIR"

export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET_NAME="$(basename "${DATASET%.*}")"

python -m Baseline.KVLink.evaluation.precompute \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --prompt-build-method "$PROMPT_BUILD_METHOD" \
  --cache-dir "$CACHE_DIR" \
  --max-documents 40 \
  --link-tokens-per-document 5

python -m Baseline.KVLink.evaluation.rag \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --prompt-build-method "$PROMPT_BUILD_METHOD" \
  --cache-dir "$CACHE_DIR" \
  --summary-output "$RESULT_DIR/kvlink_results.csv" \
  --details-output "$RESULT_DIR/${DATASET_NAME}_kvlink_details.csv" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-documents 40 \
  --link-tokens-per-document 5 \
  --qps "$QPS"
