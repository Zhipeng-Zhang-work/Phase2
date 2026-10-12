# TurboRAG-Reordered Baseline

This is an independent TurboRAG-reordered implementation for
`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`. It does not import the root
`lmcache/` package or `benchmarks/rag` Python modules.

## Layout

- `training/`: Qwen2 full-parameter FSDP fine-tuning with independent document
  attention, `<|doc_start|>`/`<|doc_end|>` boundaries, and globally reordered
  positions.
- `lmcache/`: document KV persistence, Qwen RoPE relocation, cache
  concatenation, and greedy decoding.
- `evaluation/`: independent dataset parsing, metrics, offline precompute,
  RAG evaluation, and launch script.
- `tests/`: mask and RoPE relocation tests.

## Train

Prepare JSONL rows following `Baseline/README.md`. To match the TurboRAG paper
setup, construct the file with 50% document-grounded QA rows and 50% general
instruction rows. General rows may use an empty `documents` list.

Run on an eight-GPU node:

```bash
TRAIN_DATA=/path/to/turborag_train.jsonl \
OUTPUT_DIR=/home/chzhan.chzhipen/LMCache/Baseline/checkpoints/turborag-qwen32b \
bash Baseline/TurboRAG/training/train.sh
```

The script contains model, optimizer, sequence, batch, special-token, and
checkpoint output parameters. It uses PyTorch FSDP directly and does not
require Accelerate.

## Evaluate

```bash
MODEL=/home/chzhan.chzhipen/LMCache/Baseline/checkpoints/turborag-qwen32b \
DATASET=/home/chzhan.chzhipen/LMCache/benchmarks/rag/inputs/longbench_inputs/2wikimqa_longbench.json \
PROMPT_BUILD_METHOD=QA \
bash Baseline/TurboRAG/evaluation/launch_lmcache.sh
```

Each document is independently encoded with boundary tokens. Online
evaluation loads those KV tensors, moves their post-RoPE keys from local
positions to reordered global positions, concatenates them, and computes only
the query and generated tokens.

Offline precomputation is excluded from TTFT. TTFT includes online disk cache
loading, host-to-device transfer, RoPE relocation, cache concatenation, query
prefill, and first-token selection.

The evaluator is a standalone Transformers engine, not vLLM. `QPS` controls
request pacing and requests run serially. Compare latency only against
baselines using the same standalone measurement path. Use a new `CACHE_DIR`
after changing the checkpoint.

