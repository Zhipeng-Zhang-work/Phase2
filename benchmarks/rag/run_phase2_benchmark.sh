#!/usr/bin/env bash
# Run the Phase 2 CacheBlend benchmark directly inside an allocated GPU node.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RAG_DIR="$SCRIPT_DIR"
RESULTS_DIR="${RESULTS_DIR:-$RAG_DIR/outputs/phase2_benchmark}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
SERVED_MODEL_PATH="${SERVED_MODEL_PATH:-$MODEL_NAME}"
PORT="${PORT:-8001}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
KV_STORAGE_SIZE="${KV_STORAGE_SIZE:-3000GB}"
KV_CHUNK_SIZE="${KV_CHUNK_SIZE:-256}"
RECOMPUTE_RATIO="${RECOMPUTE_RATIO:-0.16}"
PYRAMID_ENABLED="${PYRAMID_ENABLED:-false}"
QPS="${QPS:-0.25}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-30000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-5400}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
SERVER_PID=""

BLENDER_VARIANTS=(
  blender_v1 blender_v2 blender_v3 blender_v4 blender_v5 blender_v6
  blender_v7 blender_v8 blender_v9 blender_v10 blender_v11 blender_v12
  blender_v13 blender_v14 blender_v15 blender_v16 blender_v17 blender_v18
  blender_v19 blender_v20 blender_v21 blender_v22 blender_v23 blender_v24
  blender_v25 blender_v26 blender_v27 blender_v28 blender_v29 blender_v30
  blender_v31 blender_v32 blender_v33 blender_v34 blender_v35 blender_v36
  blender_v37
)

# Format: dataset path relative to benchmarks/rag|prompt construction method.
DATASET_SPECS=(
  "inputs/longbench_inputs/musique_longbench.json|QA"
  "inputs/longbench_inputs/2wikimqa_longbench.json|QA"
  "inputs/longbench_inputs/multifieldqa_zh_longbench.json|QA_ZH"
  "inputs/longbench_inputs/samsum_longbench.json|FEW_SHOT"
  "inputs/longbench_inputs/dureader_longbench.json|DUREADER"
)

usage() {
  cat <<'EOF'
Usage: bash benchmarks/rag/run_phase2_benchmark.sh

Run from a node with one usable CUDA GPU and a prepared repository .venv.
Results are resumable in benchmarks/rag/outputs/phase2_benchmark by default.

Environment overrides:
  RESULTS_DIR, MODEL_NAME, SERVED_MODEL_PATH, PORT, RECOMPUTE_RATIO,
  PYRAMID_ENABLED, QPS, MAX_MODEL_LEN, MAX_TOKENS, KV_STORAGE_SIZE.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

cleanup_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[cleanup] stopping server pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

cleanup() {
  cleanup_server
}

trap cleanup EXIT INT TERM

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

wait_for_server() {
  local server_log="$1"
  local waited=0

  while (( waited <= SERVER_READY_TIMEOUT )); do
    if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
      echo "[server] ready after ${waited}s"
      return 0
    fi
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[server] exited before readiness; recent log follows:" >&2
      tail -n 120 "$server_log" >&2 || true
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done

  echo "[server] readiness timed out after ${SERVER_READY_TIMEOUT}s" >&2
  tail -n 120 "$server_log" >&2 || true
  return 1
}

start_server() {
  local variant="$1"
  local config_path="$2"
  local server_log="$3"

  if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
    echo "Port $PORT is already serving a healthy endpoint; choose PORT=<free-port>." >&2
    return 1
  fi

  export LMCACHE_CONFIG_FILE="$config_path"
  export LMCACHE_USE_EXPERIMENTAL=False
  export LMCACHE_BLENDER_VARIANT="$variant"

  echo "[server] variant=$variant port=$PORT config=$config_path"
  (
    cd "$RAG_DIR"
    vllm serve "$SERVED_MODEL_PATH" \
      --served-model-name "$MODEL_NAME" \
      --port "$PORT" \
      --gpu-memory-utilization 0.70 \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --no-enable-chunked-prefill \
      --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
      --disable-log-requests \
      --no-enable-prefix-caching \
      --reasoning-parser qwen3 \
      --trust-remote-code
  ) >"$server_log" 2>&1 &
  SERVER_PID=$!
  wait_for_server "$server_log"
}

append_manifest() {
  local variant="$1"
  local dataset_path="$2"
  local prompt_method="$3"
  local output_file="$4"
  local debug_file="$5"
  local run_log="$6"
  local server_log="$7"

  python3 - "$RESULTS_DIR/run_manifest.csv" "$variant" "$dataset_path" \
    "$prompt_method" "$output_file" "$debug_file" "$run_log" "$server_log" \
    "$RECOMPUTE_RATIO" "$QPS" "$MAX_TOKENS" <<'PY'
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

(
    manifest_path,
    blender,
    dataset,
    prompt_build_method,
    output_csv,
    debug_csv,
    run_log,
    server_log,
    ratio,
    qps,
    max_tokens,
) = sys.argv[1:]

columns = [
    "timestamp", "blender", "dataset", "prompt_build_method", "ratio", "qps",
    "max_tokens", "output_csv", "debug_csv", "run_log", "server_log",
    "average_quality", "average_ttft", "average_tpot", "throughput",
]
patterns = {
    "average_quality": r"average_quality=([0-9.eE+-]+)",
    "average_ttft": r"average_ttft=([0-9.eE+-]+)",
    "average_tpot": r"average_tpot=([0-9.eE+-]+)",
    "throughput": r"throughput=([0-9.eE+-]+)",
}
text = Path(run_log).read_text(encoding="utf-8", errors="replace")
row = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "blender": blender,
    "dataset": dataset,
    "prompt_build_method": prompt_build_method,
    "ratio": ratio,
    "qps": qps,
    "max_tokens": max_tokens,
    "output_csv": output_csv,
    "debug_csv": debug_csv,
    "run_log": run_log,
    "server_log": server_log,
}
row.update({key: (match.group(1) if (match := re.search(pattern, text)) else "")
            for key, pattern in patterns.items()})
path = Path(manifest_path)
write_header = not path.exists()
with path.open("a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

require_file "$RAG_DIR/precompute.py"
require_file "$RAG_DIR/rag.py"
require_file "$RAG_DIR/lmcache_blend.yaml"
for spec in "${DATASET_SPECS[@]}"; do
  IFS='|' read -r dataset_path _ <<< "$spec"
  require_file "$RAG_DIR/$dataset_path"
done
for variant in "${BLENDER_VARIANTS[@]}"; do
  require_file "$REPO_ROOT/lmcache/v1/compute/blend/${variant}.py"
done

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

python3 - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("No CUDA GPU is available to PyTorch.")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

mkdir -p "$RESULTS_DIR" "$RESULTS_DIR/configs" "$RESULTS_DIR/logs"

if [[ "$PYRAMID_ENABLED" != "true" && "$PYRAMID_ENABLED" != "false" ]]; then
  echo "PYRAMID_ENABLED must be true or false, got: $PYRAMID_ENABLED" >&2
  exit 1
fi

echo "[phase2] results=$RESULTS_DIR pyramid=$PYRAMID_ENABLED ratio=$RECOMPUTE_RATIO"

for variant in "${BLENDER_VARIANTS[@]}"; do
  for spec in "${DATASET_SPECS[@]}"; do
    IFS='|' read -r dataset_path prompt_method <<< "$spec"
    dataset_name="$(basename "$dataset_path" .json)"
    run_prefix="${dataset_name}_${variant}_ratio_${RECOMPUTE_RATIO//./p}"
    output_file="$RESULTS_DIR/${run_prefix}.csv"
    debug_file="${output_file}.debug.csv"
    run_log="$RESULTS_DIR/logs/${run_prefix}.log"
    server_log="$RESULTS_DIR/logs/${run_prefix}.server.log"
    config_path="$RESULTS_DIR/configs/${run_prefix}.yaml"

    if [[ -s "$output_file" && -s "$debug_file" ]]; then
      echo "[skip] completed: $run_prefix"
      continue
    fi

    python3 - "$RAG_DIR/lmcache_blend.yaml" "$config_path" \
      "$RECOMPUTE_RATIO" "$PYRAMID_ENABLED" <<'PY'
from pathlib import Path
import re
import sys

source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
ratio = sys.argv[3]
pyramid_enabled = sys.argv[4].capitalize()
text = source_path.read_text(encoding="utf-8")
text, ratio_count = re.subn(
    r"^blend_recompute_ratios:\s*\[[^\]]*\]\s*$",
    f"blend_recompute_ratios: [{ratio}]",
    text,
    count=1,
    flags=re.MULTILINE,
)
text, pyramid_count = re.subn(
    r"^blend_pyramid_enabled:\s*.*$",
    f"blend_pyramid_enabled: {pyramid_enabled}",
    text,
    count=1,
    flags=re.MULTILINE,
)
if ratio_count != 1 or pyramid_count != 1:
    raise SystemExit("Could not set recompute ratio and pyramid configuration.")
output_path.write_text(text, encoding="utf-8")
PY

    echo "[run] variant=$variant dataset=$dataset_name method=$prompt_method"
    start_server "$variant" "$config_path" "$server_log"
    (
      cd "$RAG_DIR"
      python3 precompute.py \
        --model "$MODEL_NAME" \
        --dataset "$dataset_path" \
        --prompt-build-method "$prompt_method" \
        --kv-storage-size "$KV_STORAGE_SIZE" \
        --kv-chunk-size "$KV_CHUNK_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --base-url "$BASE_URL"
    ) 2>&1 | tee "$run_log"
    returned_end_index="$(awk '/^Precompute from / {print $5; exit}' "$run_log")"
    if [[ -z "$returned_end_index" ]]; then
      echo "Precompute returned no end index for $dataset_path" >&2
      exit 1
    fi
    (
      cd "$RAG_DIR"
      python3 rag.py \
        --qps "$QPS" \
        --model "$MODEL_NAME" \
        --dataset "$dataset_path" \
        --end-index "$returned_end_index" \
        --separator ' # # ' \
        --prompt-build-method "$prompt_method" \
        --base-url "$BASE_URL" \
        --max-tokens "$MAX_TOKENS" \
        --request-timeout "$REQUEST_TIMEOUT" \
        --output "$output_file" \
        --debug-output "$debug_file"
    ) 2>&1 | tee -a "$run_log"
    append_manifest "$variant" "$dataset_path" "$prompt_method" "$output_file" \
      "$debug_file" "$run_log" "$server_log"
    cleanup_server
  done
done

echo "[phase2] benchmark completed. Run: bash benchmarks/rag/run_phase2_judge.sh"
