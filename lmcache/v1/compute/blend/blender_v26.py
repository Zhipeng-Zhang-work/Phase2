# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Literal

# Third Party
import torch

# First Party
from lmcache.v1.compute.blend.blender_v25 import (
    _full_attention_for_kv_head,
    _KeyAwareSemanticKVShareBlender,
    _prepare_key_aware_score_tensors,
)

_EPSILON = 1e-8


def _score_documents_by_dual_attention_shift(
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
    shift_kind: Literal["absolute_log_ratio", "js_contribution"],
) -> torch.Tensor:
    """Score document tokens from current-versus-reuse attention shift."""
    (
        query_heads,
        current_key_heads,
        cached_document_keys,
        current_value_heads,
        cached_document_values,
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

        # Only reused-document Keys differ in the simulated reuse state.
        # Both attention distributions still normalize over the complete
        # causal-visible prompt before document columns are selected.
        reuse_key_head = current_key_head.clone()
        reuse_key_head[document_indices] = cached_document_keys[kv_head]
        current_attention = _full_attention_for_kv_head(
            grouped_real_queries,
            current_key_head,
            positions,
            observer_positions,
            head_dim,
        )
        reuse_attention = _full_attention_for_kv_head(
            grouped_real_queries,
            reuse_key_head,
            positions,
            observer_positions,
            head_dim,
        )
        current_document_attention = current_attention.index_select(
            2,
            document_indices,
        )
        reuse_document_attention = reuse_attention.index_select(
            2,
            document_indices,
        )

        if shift_kind == "absolute_log_ratio":
            attention_shift = current_document_attention * torch.abs(
                torch.log(
                    (current_document_attention + _EPSILON)
                    / (reuse_document_attention + _EPSILON)
                )
            )
        elif shift_kind == "js_contribution":
            a = current_document_attention
            b = reuse_document_attention
            total = a + b
            denominator = torch.where(
                total > 0,
                total,
                torch.ones_like(total),
            )
            current_term = torch.special.xlogy(
                a,
                2.0 * a / denominator,
            )
            reuse_term = torch.special.xlogy(
                b,
                2.0 * b / denominator,
            )
            attention_shift = (0.5 * (current_term + reuse_term)).clamp_min(0.0)
        else:
            raise ValueError(f"Unsupported attention shift: {shift_kind}")

        current_document_values = current_value_heads[kv_head].index_select(
            0,
            document_indices,
        )
        value_deviation = (
            (current_document_values - cached_document_values[kv_head]).abs().sum(dim=1)
        )
        scores += attention_shift.sum(dim=(0, 1)) * value_deviation

    return scores


class LMCBlender(_KeyAwareSemanticKVShareBlender):
    """V26: absolute attention log-ratio times Value mismatch."""

    variant_name = "V26-absolute-attention-log-ratio-value-mismatch"

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
        """Apply the V26 absolute attention log-ratio score."""
        return _score_documents_by_dual_attention_shift(
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
            "absolute_log_ratio",
        )
