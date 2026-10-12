# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import math

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.blender_query_aware import (
    QueryAwareDocumentBlender,
    select_document_repair_and_active_positions,
)

logger = init_logger(__name__)


def _score_documents_by_first_order_output_perturbation(
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
    """Score document tokens by the norm of their combined KV output effect.

    Query heads sharing a KV head are averaged, matching the grouped-query
    approximation used by the original attention-perturbation scorer. Each
    observer's softmax is computed over all causally visible current keys.

    Args:
        queries: RoPE-applied queries for the complete prompt.
        keys: Current RoPE-applied keys for the complete prompt.
        cached_keys: Retrieved cached keys for the reusable prefix.
        values: Current values for the complete prompt.
        cached_values: Retrieved cached values for the reusable prefix.
        observer_indices: Global query positions used to observe perturbations.
        document_indices: Reused document positions eligible for repair.
        positions: Original positions corresponding to complete-prompt rows.
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Attention head dimension.

    Returns:
        Mean first-order output-perturbation norm for each document token.

    Raises:
        ValueError: If tensor lengths or grouped-query head counts disagree.
    """
    sequence_length = keys.shape[0]
    if (
        queries.shape[0] != sequence_length
        or values.shape[0] != sequence_length
        or positions.numel() != sequence_length
    ):
        raise ValueError("First-order scoring requires complete-prompt Q/K/V")
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise ValueError("Query heads must be divisible by KV heads for GQA")
    if observer_indices.numel() == 0:
        raise ValueError("At least one observer query position is required")
    if document_indices.numel() == 0:
        raise ValueError("At least one reused document candidate is required")

    group_size = num_heads // num_kv_heads
    query_groups = (
        queries.view(sequence_length, num_kv_heads, group_size, head_dim)
        .to(torch.float32)
        .mean(dim=2)
        .permute(1, 0, 2)
    )
    current_keys = (
        keys.view(sequence_length, num_kv_heads, head_dim)
        .to(torch.float32)
        .permute(1, 0, 2)
    )
    current_values = (
        values.view(sequence_length, num_kv_heads, head_dim)
        .to(torch.float32)
        .permute(1, 0, 2)
    )

    candidate_cached_keys = (
        cached_keys.index_select(0, document_indices)
        .view(document_indices.numel(), num_kv_heads, head_dim)
        .to(torch.float32)
        .permute(1, 0, 2)
    )
    candidate_cached_values = (
        cached_values.index_select(0, document_indices)
        .view(document_indices.numel(), num_kv_heads, head_dim)
        .to(torch.float32)
        .permute(1, 0, 2)
    )
    candidate_keys = current_keys.index_select(1, document_indices)
    candidate_values = current_values.index_select(1, document_indices)
    delta_keys = candidate_keys - candidate_cached_keys
    delta_values = candidate_values - candidate_cached_values

    observer_queries = query_groups.index_select(1, observer_indices)
    logits = torch.matmul(
        observer_queries,
        current_keys.transpose(1, 2),
    ) / math.sqrt(head_dim)
    observer_positions = positions.index_select(0, observer_indices)
    causal_mask = positions.unsqueeze(0) > observer_positions.unsqueeze(1)
    logits = logits.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
    attention = torch.softmax(logits, dim=-1)

    current_outputs = torch.matmul(attention, current_values)
    candidate_attention = attention.index_select(2, document_indices)

    # Keep the signed q^T delta-k term and compute ||a*u + delta-v|| without
    # materializing an observer-by-candidate-by-head-dimension tensor.
    delta_logits = torch.matmul(
        observer_queries,
        delta_keys.transpose(1, 2),
    ) / math.sqrt(head_dim)
    value_output_dot = torch.einsum(
        "gcd,gtd->gtc",
        candidate_values,
        current_outputs,
    )
    centered_value_norm_sq = (
        candidate_values.square().sum(dim=-1).unsqueeze(1)
        - 2.0 * value_output_dot
        + current_outputs.square().sum(dim=-1).unsqueeze(-1)
    )
    delta_value_norm_sq = delta_values.square().sum(dim=-1).unsqueeze(1)
    candidate_delta_value_dot = (candidate_values * delta_values).sum(dim=-1)
    output_delta_value_dot = torch.einsum(
        "gcd,gtd->gtc",
        delta_values,
        current_outputs,
    )
    centered_delta_value_dot = (
        candidate_delta_value_dot.unsqueeze(1) - output_delta_value_dot
    )

    norm_sq = (
        delta_logits.square() * centered_value_norm_sq
        + delta_value_norm_sq
        + 2.0 * delta_logits * centered_delta_value_dot
    )
    norm_sq = norm_sq.clamp_min(0.0)
    edge_scores = candidate_attention * torch.sqrt(norm_sq)
    document_scores = edge_scores.mean(dim=1).mean(dim=0)
    if document_scores.numel() != document_indices.numel():
        raise AssertionError("Document score count does not match candidates")
    return document_scores


class LMCBlender(QueryAwareDocumentBlender):
    """V17: full-query signed first-order KV output-perturbation repair."""

    variant_name = "V17-full-query-vector-preserving-first-order-repair"

    def _observer_indices(self) -> torch.Tensor:
        """Use every prompt position as an output-observation query."""
        assert self.full_prompt_length is not None
        assert self.prompt_spans is not None
        return torch.arange(
            self.full_prompt_length,
            dtype=torch.int64,
            device=self.prompt_spans.document_indices.device,
        )

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: Optional[torch.Tensor],
        attn_metadata: LMCAttnMetadata,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        LMCAttnMetadata,
    ]:
        """Select document KV repairs at the configured checking layer."""
        old_k, old_v = self.gpu_connector.get_kv(layer_id)
        if attn_output is None:
            attn_output = torch.empty(q.shape, dtype=q.dtype, device=q.device)

        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0],
                device=q.device,
                dtype=torch.int64,
            )
        active_positions = self.metadata.positions
        assert active_positions is not None
        assert self.full_prompt_length is not None
        assert self.cached_token_count is not None
        if old_k.shape[0] != self.cached_token_count:
            raise ValueError(
                "Retrieved layerwise cache length does not match the reusable "
                f"prefix: {old_k.shape[0]} != {self.cached_token_count}"
            )

        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        q, k = layer.self_attn.rotary_emb(active_positions, q, k)

        check_layer = self.common_metadata.check_layers[0]
        if layer_id < check_layer:
            self._copy_active_rows_to_cache(old_k, k, active_positions)
            self._copy_active_rows_to_cache(old_v, v, active_positions)
            return q, k, v, residual, attn_output, attn_metadata

        if layer_id == check_layer:
            expected_positions = torch.arange(
                self.full_prompt_length,
                dtype=torch.int64,
                device=q.device,
            )
            if not torch.equal(active_positions, expected_positions):
                raise ValueError(
                    "The checking layer requires complete prompt Q/K positions"
                )
            document_indices, mandatory_indices = self._candidate_and_mandatory_indices(
                q.device
            )
            observer_indices = self._observer_indices()
            attn_layer = self.layerwise_model.vllm_attn_layers[layer_id]
            scores = _score_documents_by_first_order_output_perturbation(
                q,
                k,
                old_k,
                v,
                old_v,
                observer_indices,
                document_indices,
                active_positions,
                attn_layer.num_heads,
                attn_layer.num_kv_heads,
                attn_layer.head_size,
            )

            assert self.common_metadata.recomp_ratios is not None
            ratio = self.common_metadata.recomp_ratios[0]
            selected_documents, active_positions = (
                select_document_repair_and_active_positions(
                    scores,
                    document_indices,
                    mandatory_indices,
                    ratio,
                    self.full_prompt_length,
                )
            )
            repair_budget = selected_documents.numel()

            local_active = torch.searchsorted(
                self.metadata.positions,
                active_positions,
            )
            active_k = k[local_active]
            active_v = v[local_active]
            self._copy_active_rows_to_cache(old_k, active_k, active_positions)
            self._copy_active_rows_to_cache(old_v, active_v, active_positions)
            full_k = self._assemble_full_keys(
                old_k,
                active_k,
                active_positions,
                self.full_prompt_length,
            )
            full_v = self._assemble_full_keys(
                old_v,
                active_v,
                active_positions,
                self.full_prompt_length,
            )

            q = q[local_active]
            residual = residual[local_active]
            self.metadata.imp_indices = selected_documents
            self.metadata.positions = active_positions
            attn_output = attn_output[: active_positions.numel()]
            attn_metadata.update_from_top_indices(active_positions)
            logger.info(
                "%s selected %d/%d reused document tokens; active=%d",
                self.variant_name,
                repair_budget,
                document_indices.numel(),
                active_positions.numel(),
            )
            return q, full_k, full_v, residual, attn_output, attn_metadata

        self._copy_active_rows_to_cache(old_k, k, active_positions)
        self._copy_active_rows_to_cache(old_v, v, active_positions)
        full_k = self._assemble_full_keys(
            old_k,
            k,
            active_positions,
            self.full_prompt_length,
        )
        full_v = self._assemble_full_keys(
            old_v,
            v,
            active_positions,
            self.full_prompt_length,
        )
        return q, full_k, full_v, residual, attn_output, attn_metadata
