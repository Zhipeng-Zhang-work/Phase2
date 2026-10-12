# SPDX-License-Identifier: Apache-2.0
# Standard
import math

# Third Party
import torch

# First Party
from lmcache.v1.compute.blend.blender_v21 import _SemanticKVShareBlender


def _score_documents_with_per_query_head_attention(
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
    """Score documents with per-query-head softmax and aligned V deviation.

    Every real query head retains its own logits and performs softmax over the
    complete causally visible key domain for all Instruction+Question observer
    rows before reused-document columns are extracted.

    Args:
        queries: RoPE-applied queries for the complete prompt.
        keys: Current RoPE-applied keys for the complete prompt.
        values: Current values for the complete prompt.
        cached_values: Retrieved cached values for the reusable prefix.
        observer_indices: Instruction and question positions.
        document_indices: Reused document positions eligible for repair.
        positions: Original positions corresponding to complete-prompt rows.
        num_heads: Number of real query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Attention head dimension.

    Returns:
        One head-aligned attention-times-Value-deviation score per reused
        document token.

    Raises:
        ValueError: If tensor lengths, observer sets, candidates, or GQA head
            counts are invalid.
    """
    sequence_length = keys.shape[0]
    if (
        queries.shape[0] != sequence_length
        or values.shape[0] != sequence_length
        or positions.numel() != sequence_length
    ):
        raise ValueError("Semantic scoring requires complete-prompt Q/K/V")
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise ValueError("Query heads must be divisible by KV heads for GQA")
    if observer_indices.numel() == 0:
        raise ValueError("Instruction+Question observer set cannot be empty")
    if document_indices.numel() == 0:
        raise ValueError("Reused document candidate set cannot be empty")

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
    observer_positions = positions.index_select(0, observer_indices)
    attention_mass_per_kv_head = torch.zeros(
        (num_kv_heads, document_indices.numel()),
        dtype=torch.float32,
        device=queries.device,
    )

    # V24 does not aggregate GQA query vectors before softmax.
    # Each real query head computes attention against its shared KV head,
    # and attention probabilities are aggregated only after softmax.
    for kv_head in range(num_kv_heads):
        query_head_start = kv_head * group_size
        query_head_end = query_head_start + group_size
        grouped_real_queries = query_heads[
            query_head_start:query_head_end
        ].index_select(1, observer_indices)
        current_key_head = key_heads[kv_head]
        logits = torch.matmul(
            grouped_real_queries,
            current_key_head.transpose(0, 1),
        ) / math.sqrt(head_dim)
        causal_mask = positions.view(1, 1, -1) > observer_positions.view(1, -1, 1)
        logits = logits.masked_fill(causal_mask, float("-inf"))

        # Softmax remains over the complete causal-visible prompt. Only
        # afterward are reused-document candidate columns extracted.
        full_attention = torch.softmax(logits, dim=-1)
        document_attention = full_attention.index_select(2, document_indices)
        attention_mass_per_kv_head[kv_head] = document_attention.sum(dim=(0, 1))

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
    value_deviation = (
        (current_document_values - cached_document_values).abs().sum(dim=2)
    )
    document_scores = (attention_mass_per_kv_head * value_deviation).sum(dim=0)
    if document_scores.numel() != document_indices.numel():
        raise AssertionError("Document score count does not match candidates")
    return document_scores


class LMCBlender(_SemanticKVShareBlender):
    """V24: V23 scoring with per-query-head GQA attention softmax."""

    variant_name = "V24-v23-per-query-head-gqa-attention"

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
        """Apply per-query-head attention with KV-head-aligned deviation."""
        return _score_documents_with_per_query_head_attention(
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
