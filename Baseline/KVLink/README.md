# KVLink Baseline

This is an independent implementation for
`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`. It does not import the root
`lmcache/` package or `benchmarks/rag` Python modules.

## Layout

- `training/`: Qwen2 full-parameter FSDP fine-tuning with KVLink link tokens
  and an independent-document 4D attention mask.
- `lmcache/`: document KV persistence, Qwen RoPE relocation, cache
  concatenation, online link-token computation, and greedy decoding.
- `evaluation/`: independent dataset parsing, metrics, offline precompute,
  RAG evaluation, and launch script.
- `tests/`: mask and RoPE relocation tests.
- `upstream/`: unmodified public KVLink repository retained only as reference.

## Train

Prepare JSONL rows following `Baseline/README.md`. The file should reproduce
the intended task mixture before training. For the paper-style KVLink mixture,
use 20% continuation text, 30% general instruction data, 25% memory
instruction data, 10% ordinary QA, 10% memory QA, and 5% summarization.

Run on an eight-GPU node:

```bash
TRAIN_DATA=/path/to/kvlink_train.jsonl \
OUTPUT_DIR=/home/chzhan.chzhipen/LMCache/Baseline/checkpoints/kvlink-qwen32b \
bash Baseline/KVLink/training/train.sh
```

The script defines all model, optimizer, sequence, batch, link-token, and
checkpoint output parameters. It uses PyTorch FSDP directly and does not
require Accelerate.

## Evaluate

```bash
MODEL=/home/chzhan.chzhipen/LMCache/Baseline/checkpoints/kvlink-qwen32b \
DATASET=/home/chzhan.chzhipen/LMCache/benchmarks/rag/inputs/longbench_inputs/2wikimqa_longbench.json \
PROMPT_BUILD_METHOD=QA \
bash Baseline/KVLink/evaluation/launch_lmcache.sh
```

`precompute.py` runs before `rag.py`, so document precomputation is excluded
from TTFT. TTFT starts immediately before online disk cache loading and
includes cache loading, host-to-device transfer, RoPE relocation,
concatenation, online link-token computation, query prefill, and first-token
selection.

The evaluator is a standalone Transformers engine, not vLLM. `QPS` controls
request pacing; requests are executed serially because the engine does not
implement a continuous batching scheduler. Compare these latency results only
against baselines run through the same standalone measurement path.

Use a new `CACHE_DIR` after changing the model checkpoint. This implementation
covers ordinary KVLink and intentionally does not include the optional
LLMLingua/ANLLM document-compression experiment.

