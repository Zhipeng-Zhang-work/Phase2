# Phase 2 Track B: CacheBlend RAG Benchmark

This repository is the runnable deliverable for Huawei Phase 2. It uses LMCache
and vLLM to evaluate CacheBlend strategies on long-context RAG workloads, with
answer quality, TTFT, TPOT, and request throughput as the primary metrics.

The delivery is intentionally limited to one CacheBlend benchmark workflow: one
benchmark entry point, one local LLM judge entry point, and the RAG code,
configuration, and LongBench inputs they require. C2KV, CacheTune, SpecCache,
selective-recompute audits, historical Slurm scripts, and historical results are
outside this delivery.

## Workflow

```text
LongBench inputs
    |
    v
run_phase2_benchmark.sh
  starts vLLM + LMCache, precomputes KV caches, and runs RAG
    |
    v
outputs/phase2_benchmark/
  CSVs, debug CSVs, logs, generated configs, and run_manifest.csv
    |
    v
run_phase2_judge.sh
  local LLM evaluation with per-request details and aggregate metrics
```

The benchmark runs `blender_v1` through `blender_v37`. Each Blender is run with a
fresh server on each of five LongBench datasets, giving 185 resumable experiment
units.

## Layout

```text
benchmarks/rag/
├── run_phase2_benchmark.sh  # Main benchmark entry point
├── run_phase2_judge.sh      # Local LLM judge entry point
├── phase2_judge.py          # Judge implementation
├── lmcache_blend.yaml       # Default LMCache/CacheBlend configuration
├── precompute.py            # Independent KV-cache precomputation
├── rag.py                   # Online RAG requests and static metrics
├── utils.py
├── output_processing.py
├── requirements.txt
└── inputs/longbench_inputs/ # The only retained input datasets
```

The retained datasets and their evaluation modes are:

| Dataset | Prompt mode | Static metric |
| --- | --- | --- |
| `musique_longbench.json` | `QA` | F1 |
| `2wikimqa_longbench.json` | `QA` | F1 |
| `multifieldqa_zh_longbench.json` | `QA_ZH` | F1 |
| `samsum_longbench.json` | `FEW_SHOT` | ROUGE-L |
| `dureader_longbench.json` | `DUREADER` | ROUGE-L |

## Prerequisites

Run the scripts on a Linux CUDA GPU node. Do not run them on a login node and do
not submit them with `sbatch`; if your cluster uses a scheduler, first obtain a
GPU allocation and execute the scripts inside that allocation.

Prepare a Python environment with CUDA-compatible PyTorch, vLLM, and LMCache.
Both entry points automatically activate a repository-root `.venv` when present.
Install the RAG dependencies with the following commands after installing the
GPU-specific stack:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch
uv pip install -e . --no-build-isolation
uv pip install -r benchmarks/rag/requirements.txt
```

The default benchmark model is `Qwen/Qwen3-32B`; the default judge model is
`google/gemma-3-27b-it`. Both scripts use the local Hugging Face cache by default,
so ensure that both models are available on the GPU node. To permit downloads,
explicitly set `HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0`.

## Run the Benchmark

From the repository root on an allocated GPU node:

```bash
bash benchmarks/rag/run_phase2_benchmark.sh
```

The script performs the following work:

1. Validates CUDA availability, five datasets, and all 37 Blender implementations.
2. Produces a temporary LMCache configuration for each Blender-dataset unit.
3. Starts vLLM with LMCache, precomputes independent KV caches, and runs online RAG.
4. Stops the server for that unit and writes CSVs, debug CSVs, logs, and a manifest.

The default result location is `benchmarks/rag/outputs/phase2_benchmark/`. Re-run
the same command after an interruption to skip units that already have both their
result CSV and debug CSV.

The default configuration uses Qwen3-32B, recompute ratio `0.16`, QPS `0.25`, a
maximum context length of `30000`, and a maximum generation length of `2048`.
**Pyramid is disabled by default.**

Common overrides:

```bash
# Use a local model path and a different port.
SERVED_MODEL_PATH=/path/to/Qwen3-32B PORT=8010 \
  bash benchmarks/rag/run_phase2_benchmark.sh

# Use an isolated result directory and explicitly enable Pyramid.
RESULTS_DIR=/path/to/phase2-run PYRAMID_ENABLED=true \
  bash benchmarks/rag/run_phase2_benchmark.sh

# Change the experimental load.
RECOMPUTE_RATIO=0.20 QPS=0.10 MAX_TOKENS=1024 \
  bash benchmarks/rag/run_phase2_benchmark.sh
```

View the supported entry-point options with:

```bash
bash benchmarks/rag/run_phase2_benchmark.sh --help
```

## Run the LLM Judge

After the benchmark completes, or when the target directory already contains a
valid `run_manifest.csv`, run the judge on a GPU node with the local judge model:

```bash
bash benchmarks/rag/run_phase2_judge.sh
```

The judge reads the benchmark outputs, removes reasoning markup before scoring,
and records empty answers as zero without sending them to the judge model. It
writes the following files:

```text
benchmarks/rag/outputs/phase2_benchmark/judge/
├── judge_details.csv  # Static scores, judge scores, and text audit fields
└── judge_summary.csv  # Metrics grouped by Blender, dataset, ratio, and prompt mode
```

The judge is resumable: rows with an existing `source_key` are skipped. To judge a
different benchmark directory or use another local judge model:

```bash
BENCHMARK_DIR=/path/to/phase2-run \
JUDGE_MODEL=/path/to/local-judge-model \
MAX_NEW_TOKENS=2048 \
  bash benchmarks/rag/run_phase2_judge.sh
```

## Outputs and Interpretation

- Benchmark CSV: static quality, TTFT, TPOT, generation time, and token counts for each request.
- `*.debug.csv`: request prompts, raw generations, cleaned outputs, and references for auditing.
- `run_manifest.csv`: paths and aggregate quality/latency metrics for each completed unit.
- `judge_details.csv`: local LLM semantic evaluation for every request.
- `judge_summary.csv`: judge accuracy, mean judge score, static quality, mean TTFT, and mean TPOT.

File-existence checks, syntax checks, and static validation establish only that the
delivery structure is valid. Use the generated result CSVs from a complete target
GPU run for any conclusion about performance, quality, or model behavior.

## License

The LMCache source code is licensed under the [Apache License 2.0](LICENSE).
