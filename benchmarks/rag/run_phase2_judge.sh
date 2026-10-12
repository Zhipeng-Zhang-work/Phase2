#!/usr/bin/env bash
# Run the Phase 2 local LLM judge directly inside an allocated GPU node.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
BENCHMARK_DIR="${BENCHMARK_DIR:-$SCRIPT_DIR/outputs/phase2_benchmark}"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemma-3-27b-it}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"

usage() {
  cat <<'EOF'
Usage: bash benchmarks/rag/run_phase2_judge.sh

Judge the benchmark rows in benchmarks/rag/outputs/phase2_benchmark by default.
Set BENCHMARK_DIR to judge a different benchmark output directory.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$BENCHMARK_DIR/run_manifest.csv" ]]; then
  echo "Missing benchmark manifest: $BENCHMARK_DIR/run_manifest.csv" >&2
  exit 1
fi
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

cd "$REPO_ROOT"
python3 "$SCRIPT_DIR/phase2_judge.py" \
  --benchmark-dir "$BENCHMARK_DIR" \
  --judge-model "$JUDGE_MODEL" \
  --max-new-tokens "$MAX_NEW_TOKENS"
