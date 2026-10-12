# SPDX-License-Identifier: Apache-2.0
# Standard
import math

# Third Party
import torch

# First Party
from lmcache.v1.compute.blend.blender_v21 import _SemanticKVShareBlender

_EPSILON = 1e-8


def _score_documents_by_relative_value_mismatch(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    cached_values: torch.Tensor,
    observer_indices: torch.Tensor,
    document_indices: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Return V28 per-KV-head relative Value mismatch scores."""
    sequence_length = keys.shape[0]
    if (
        queries.shape[0] != sequence_length
        or values.shape[0] != sequence_length
        or positions.numel() != sequence_length
    ):
        raise ValueError("Relative-Value scoring requires complete-prompt Q/K/V")
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise ValueError("Query heads must be divisible by KV heads for GQA")
    if observer_indices.numel() == 0:
        raise ValueError("Instruction+Question observer set cannot be empty")
    if document_indices.numel() == 0:
        raise ValueError("Reused document candidate set cannot be empty")
    if torch.any(document_indices >= cached_values.shape[0]):
        raise ValueError("Every document candidate must have a retrieved cached Value")

    group_size = num_heads // num_kv_heads
    query_heads = (
        queries.view(sequence_length, num_heads, head_dim)
        .to(torch.float32)
        .permute(1, 0, 2)
    )
    key_heads = (
        keys.view(sequence_length, num_kv_heads, head_dim)
        .to(torch.float32)
        .permute(1, 0, 2)
    )
    current_document_values = (
        values.index_select(0, document_indices)
        .view(document_indices.numel(), num_kv_heads, head_dim)
        .to(torch.float32)
        .permute(1, 0, 2)
    )
    cached_document_values = (
        cached_values.index_select(0, document_indices)
        .view(document_indices.numel(), num_kv_heads, head_dim)
        .to(torch.float32)
        .permute(1, 0, 2)
    )
    value_difference_norm = (
        (current_document_values - cached_document_values).abs().sum(dim=2)
    )
    value_scale = current_document_values.abs().sum(
        dim=2
    ) + cached_document_values.abs().sum(dim=2)
    relative_value_mismatch = value_difference_norm / (value_scale + _EPSILON)

    observer_positions = positions.index_select(0, observer_indices)
    attention_mass_per_kv_head = torch.zeros(
        (num_kv_heads, document_indices.numel()),
        dtype=torch.float32,
        device=queries.device,
    )
    for kv_head in range(num_kv_heads):
        query_head_start = kv_head * group_size
        query_head_end = query_head_start + group_size
        grouped_real_queries = query_heads[
            query_head_start:query_head_end
        ].index_select(1, observer_indices)
        logits = torch.matmul(
            grouped_real_queries,
            key_heads[kv_head].transpose(0, 1),
        ) / math.sqrt(head_dim)
        causal_mask = positions.view(1, 1, -1) > observer_positions.view(1, -1, 1)
        logits = logits.masked_fill(causal_mask, float("-inf"))
        full_attention = torch.softmax(logits, dim=-1)
        document_attention = full_attention.index_select(2, document_indices)
        attention_mass_per_kv_head[kv_head] = document_attention.sum(dim=(0, 1))

    return (attention_mass_per_kv_head * relative_value_mismatch).sum(dim=0)


class LMCBlender(_SemanticKVShareBlender):
    """V28: V24 attention weighted by relative Value mismatch."""

    variant_name = "V28-relative-value-mismatch"

    def _score_documents(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        cached_values: torch.Tensor,
        observer_indices: torch.Tensor,
        document_indices: torch.Tensor,
        positions: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Apply the V28 relative Value mismatch score."""
        return _score_documents_by_relative_value_mismatch(
            queries,
            keys,
            values,
            cached_values,
            observer_indices,
            document_indices,
            positions,
            num_heads,
            num_kv_heads,
            head_dim,
        )
