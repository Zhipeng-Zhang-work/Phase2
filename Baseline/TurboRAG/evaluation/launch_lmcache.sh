#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/chzhan.chzhipen/LMCache}"
MODEL="${MODEL:-$REPO_ROOT/Baseline/checkpoints/turborag-qwen32b}"
DATASET="${DATASET:-$REPO_ROOT/benchmarks/rag/inputs/longbench_inputs/2wikimqa_longbench.json}"
PROMPT_BUILD_METHOD="${PROMPT_BUILD_METHOD:-QA}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/Baseline/TurboRAG/evaluation/cache}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/Baseline/TurboRAG/evaluation/results}"
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

python -m Baseline.TurboRAG.evaluation.precompute \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --prompt-build-method "$PROMPT_BUILD_METHOD" \
  --cache-dir "$CACHE_DIR" \
  --max-documents 40

python -m Baseline.TurboRAG.evaluation.rag \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --prompt-build-method "$PROMPT_BUILD_METHOD" \
  --cache-dir "$CACHE_DIR" \
  --summary-output "$RESULT_DIR/turborag_results.csv" \
  --details-output "$RESULT_DIR/${DATASET_NAME}_turborag_details.csv" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-documents 40 \
  --qps "$QPS"
