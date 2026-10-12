#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/chzhan.chzhipen/LMCache}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"
TRAIN_DATA="${TRAIN_DATA:-$REPO_ROOT/Baseline/data/turborag_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/Baseline/checkpoints/turborag-qwen32b}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"

export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
  --standalone \
  --nproc-per-node "$NPROC_PER_NODE" \
  --module Baseline.TurboRAG.training.train \
  --model "$MODEL" \
  --train-data "$TRAIN_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --max-length 4096 \
  --max-documents 40 \
  --micro-batch-size 1 \
  --global-batch-size 256 \
  --steps 6000 \
  --warmup-steps 600 \
  --learning-rate 1e-5 \
  --weight-decay 0.1 \
  --max-grad-norm 1.0 \
  --seed 0 \
  --log-every 10
