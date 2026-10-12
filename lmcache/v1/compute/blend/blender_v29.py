# SPDX-License-Identifier: Apache-2.0
# Standard
import math

# Third Party
import torch

# First Party
from lmcache.v1.compute.blend.blender_v25 import (
    _full_attention_for_kv_head,
    _KeyAwareSemanticKVShareBlender,
    _prepare_key_aware_score_tensors,
)


def _score_documents_by_key_induced_output_impact(
    queries: torch.Tensor,
    keys: torch.Tensor,
    cached_keys: torch.Tensor,
    values: torch.Tensor,
    cached_values: torch.Tensor,
    observer_indices: torch.Tensor,
    document_indices: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Return V29 Key-routing output-impact scores for reused documents."""
    (
        query_heads,
        current_key_heads,
        cached_document_keys,
        current_value_heads,
        _,
        observer_positions,
        group_size,
    ) = _prepare_key_aware_score_tensors(
        queries,
        keys,
        cached_keys,
        values,
        cached_values,
        observer_indices,
        document_indices,
        positions,
        num_heads,
        num_kv_heads,
        head_dim,
    )
    scores = torch.zeros(
        document_indices.numel(),
        dtype=torch.float32,
        device=queries.device,
    )

    for kv_head in range(num_kv_heads):
        query_head_start = kv_head * group_size
        query_head_end = query_head_start + group_size
        grouped_real_queries = query_heads[
            query_head_start:query_head_end
        ].index_select(1, observer_indices)
        current_key_head = current_key_heads[kv_head]
        current_value_head = current_value_heads[kv_head]
        full_attention = _full_attention_for_kv_head(
            grouped_real_queries,
            current_key_head,
            positions,
            observer_positions,
            head_dim,
        )
        attention_outputs = torch.matmul(full_attention, current_value_head)
        document_attention = full_attention.index_select(2, document_indices)

        current_document_keys = current_key_head.index_select(0, document_indices)
        delta_keys = current_document_keys - cached_document_keys[kv_head]
        delta_logits = torch.matmul(
            grouped_real_queries,
            delta_keys.transpose(0, 1),
        ) / math.sqrt(head_dim)

        current_document_values = current_value_head.index_select(
            0,
            document_indices,
        )
        document_value_norm_sq = current_document_values.square().sum(dim=1)
        output_norm_sq = attention_outputs.square().sum(dim=2)
        output_value_dot = torch.matmul(
            attention_outputs,
            current_document_values.transpose(0, 1),
        )
        leverage_sq = (
            document_value_norm_sq.view(1, 1, -1)
            + output_norm_sq.unsqueeze(2)
            - 2.0 * output_value_dot
        ).clamp_min(0.0)
        value_leverage = torch.sqrt(leverage_sq)
        scores += (document_attention * delta_logits.abs() * value_leverage).sum(
            dim=(0, 1)
        )

    return scores


class LMCBlender(_KeyAwareSemanticKVShareBlender):
    """V29: Key-induced first-order attention-output impact."""

    variant_name = "V29-key-induced-output-impact"

    def _score_documents_key_aware(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        cached_keys: torch.Tensor,
        values: torch.Tensor,
        cached_values: torch.Tensor,
        observer_indices: torch.Tensor,
        document_indices: torch.Tensor,
        positions: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Apply the V29 Key-induced output-impact score."""
        return _score_documents_by_key_induced_output_impact(
            queries,
            keys,
            cached_keys,
            values,
            cached_values,
            observer_indices,
            document_indices,
            positions,
            num_heads,
            num_kv_heads,
            head_dim,
        )
