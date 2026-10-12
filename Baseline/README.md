# Independent KV-Reuse Baselines

This directory contains two isolated baselines for
`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`:

- `KVLink`: document KV reuse with trainable link tokens.
- `TurboRAG`: document KV reuse with independent document attention and
  reordered global positions.

Each baseline owns its complete `training/` and `evaluation/` implementation.
Neither implementation imports `lmcache/` or `benchmarks/rag/`. The original
KVLink repository is preserved under `KVLink/upstream/` for reference only and
is not on either runtime import path.

## Data contracts

Training consumes JSON Lines. Each row must contain:

```json
{
  "documents": ["document one", "document two"],
  "question": "The question",
  "answer": "The reference answer"
}
```

Evaluation consumes the existing benchmark JSON format:

```json
[
  {
    "ctxs": [{"title": "title", "text": "text"}],
    "question": "question",
    "answers": ["reference answer"]
  }
]
```

For summarization/few-shot data, `ctxs[*].text`, `question`, and `answers` are
used directly.

## Isolation guarantee

The two baseline directories intentionally duplicate prompt, metric, cache,
RoPE, generation, and training-mask logic. This avoids accidental behavior
changes when CacheBlend or the other baseline is modified. The only shared
inputs are model checkpoints and dataset files supplied through command-line
arguments.

