# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Literal, Optional
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


def _grouped_semantic_attention_and_value_deviation(
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
    query_reduction: Literal["mean", "max"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute grouped-query attention and per-KV-head value deviation.

    Attention is normalized over every causally visible key in the complete
    prompt. Reused-document columns are extracted only after that softmax.

    Args:
        queries: RoPE-applied queries for the complete prompt.
        keys: Current RoPE-applied keys for the complete prompt.
        values: Current values for the complete prompt.
        cached_values: Retrieved cached values for the reusable prefix.
        observer_indices: Instruction and question positions.
        document_indices: Reused document positions eligible for repair.
        positions: Original positions corresponding to complete-prompt rows.
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Attention head dimension.
        query_reduction: GQA query-vector reduction, either mean or maximum.

    Returns:
        Document attention with shape ``[G, T, D_doc]`` and Value L1
        deviation with shape ``[G, D_doc]``.

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
    query_heads = queries.view(sequence_length, num_heads, head_dim).to(torch.float32)
    grouped_queries = query_heads.view(
        sequence_length,
        num_kv_heads,
        group_size,
        head_dim,
    )
    if query_reduction == "mean":
        grouped_queries = grouped_queries.mean(dim=2)
    elif query_reduction == "max":
        grouped_queries = grouped_queries.max(dim=2).values
    else:
        raise ValueError(f"Unsupported GQA query reduction: {query_reduction}")
    grouped_queries = grouped_queries.permute(1, 0, 2)

    key_heads = (
        keys.view(sequence_length, num_kv_heads, head_dim)
        .to(torch.float32)
        .permute(1, 0, 2)
    )
    observer_queries = grouped_queries.index_select(1, observer_indices)
    logits = torch.matmul(
        observer_queries,
        key_heads.transpose(1, 2),
    ) / math.sqrt(head_dim)
    observer_positions = positions.index_select(0, observer_indices)
    causal_mask = positions.unsqueeze(0) > observer_positions.unsqueeze(1)
    logits = logits.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

    full_attention = torch.softmax(logits, dim=-1)
    document_attention = full_attention.index_select(2, document_indices)

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
    return document_attention, value_deviation


def _score_documents_with_cross_head_product(
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
    query_reduction: Literal["mean", "max"],
) -> torch.Tensor:
    """Return V10/V11 cross-head-product scores for reused documents."""
    document_attention, value_deviation = (
        _grouped_semantic_attention_and_value_deviation(
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
            query_reduction,
        )
    )
    cumulative_attention = document_attention.sum(dim=(0, 1))
    cumulative_value_deviation = value_deviation.sum(dim=0)
    document_scores = cumulative_attention * cumulative_value_deviation
    if document_scores.numel() != document_indices.numel():
        raise AssertionError("Document score count does not match candidates")
    return document_scores


class _SemanticKVShareBlender(QueryAwareDocumentBlender):
    """Shared full-prompt plumbing for semantic KVShare-style scoring."""

    include_instruction_observers = True
    variant_name = "semantic-kvshare-document-repair"

    def _observer_indices(self) -> torch.Tensor:
        """Return the semantic Instruction+Question observer positions."""
        assert self.prompt_spans is not None
        return torch.cat(
            (
                self.prompt_spans.instruction_indices,
                self.prompt_spans.question_indices,
            )
        )

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
        """Return one variant-specific score per reused document token."""
        raise NotImplementedError

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
        """Select document repairs at the configured checking layer."""
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
            scores = self._score_documents(
                q,
                k,
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


class LMCBlender(_SemanticKVShareBlender):
    """V21: V10 scoring with Instruction+Question semantic observers."""

    variant_name = "V21-v10-semantic-instruction-question-observer"

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
        """Apply V10's mean-GQA cross-head-product score to I+Q observers."""
        # V21 replaces the old last-N reusable-prefix observation window with
        # the semantic Instruction+Question observer.
        #
        # The old last-N window did not observe the current user query because
        # the legacy CacheBlend path only received the reusable cached prefix.
        return _score_documents_with_cross_head_product(
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
            "mean",
        )
