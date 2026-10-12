# SPDX-License-Identifier: Apache-2.0

# KVRecompute

`kvrecompute` is a small training scaffold for learning a token-level recompute
scorer for LMCache blending.

The design follows the *workflow shape* of `kvzap`, but the task is different:

- `kvzap` predicts token/head importance for KV compression
- `kvrecompute` predicts token-level **recompute benefit** for LMCache blending

## Goal

For one request and one selected layer, we want a model that predicts a scalar
score for every token:

```text
score_t ~= "how much final answer loss would improve if token t were recomputed"
```

At inference time, the online blender can sort these scores and choose:

```text
top_indices = topk(score)
```

## Feature Design

The first version only uses features that are directly available inside
`blender.process_qkv(...)`:

- `||k - old_k||`
- `||v - old_v||`
- `||residual||`
- `delta_pos`

where:

- `old_k`, `old_v` are reused cached tensors after RoPE position remapping
- `delta_pos = new_position - cached_position`

## Teacher Label Design

Unlike `kvzap`, the label here should reflect **recompute utility**, not
compression utility.

The intended token-level teacher score is:

```text
y_t = CE_reuse(answer) - CE_recompute_only_t(answer)
```

where:

- `CE_reuse(answer)` is the cross-entropy on the gold answer when reusing cache
- `CE_recompute_only_t(answer)` is the same loss after recomputing only token `t`

This repository scaffold does **not** generate those labels yet. Instead, it
expects a precomputed teacher dataset saved with `torch.save`.

## Expected Teacher Dataset Format

Save a dictionary with:

```python
{
    "features": torch.Tensor,  # [num_tokens, 4]
    "labels": torch.Tensor,    # [num_tokens]
    "metadata": {...},         # optional
}
```

## Training

```bash
python -m kvrecompute.train \
  --dataset /path/to/recompute_teacher.pt \
  --output-dir /path/to/output
```

This trains a small MLP regressor and saves `token_mlp.pt`.

## Next Step

The missing piece is a Hugging Face based offline collector that:

1. reconstructs `old_k/old_v` without vLLM,
2. computes token-level features,
3. evaluates `CE_reuse` and `CE_recompute_only_t`,
4. writes the teacher dataset expected by this package.
