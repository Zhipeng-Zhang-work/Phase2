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
    select_document_repair_and_active_positions,
)
from lmcache.v1.compute.blend.blender_v21 import _SemanticKVShareBlender

logger = init_logger(__name__)


def _prepare_key_aware_score_tensors(
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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Validate inputs and expose float32 tensors for key-aware scoring."""
    sequence_length = keys.shape[0]
    if (
        queries.shape[0] != sequence_length
        or values.shape[0] != sequence_length
        or positions.numel() != sequence_length
    ):
        raise ValueError("Key-aware scoring requires complete-prompt Q/K/V")
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise ValueError("Query heads must be divisible by KV heads for GQA")
    if observer_indices.numel() == 0:
        raise ValueError("Instruction+Question observer set cannot be empty")
    if document_indices.numel() == 0:
        raise ValueError("Reused document candidate set cannot be empty")
    if torch.any(document_indices >= cached_keys.shape[0]):
        raise ValueError("Every document candidate must have a retrieved cached Key")
    if torch.any(document_indices >= cached_values.shape[0]):
        raise ValueError("Every document candidate must have a retrieved cached Value")

    group_size = num_heads // num_kv_heads
    query_heads = (
        queries.view(sequence_length, num_heads, head_dim)
        .to(torch.float32)
        .permute(1, 0, 2)
    )
    current_key_heads = (
        keys.view(sequence_length, num_kv_heads, head_dim)
        .to(torch.float32)
        .permute(1, 0, 2)
    )
    current_value_heads = (
        values.view(sequence_length, num_kv_heads, head_dim)
        .to(torch.float32)
        .permute(1, 0, 2)
    )
    cached_document_keys = (
        cached_keys.index_select(0, document_indices)
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
    observer_positions = positions.index_select(0, observer_indices)
    return (
        query_heads,
        current_key_heads,
        cached_document_keys,
        current_value_heads,
        cached_document_values,
        observer_positions,
        group_size,
    )


def _full_attention_for_kv_head(
    grouped_real_queries: torch.Tensor,
    current_key_head: torch.Tensor,
    positions: torch.Tensor,
    observer_positions: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Compute real-query-head attention over the complete causal prompt."""
    logits = torch.matmul(
        grouped_real_queries,
        current_key_head.transpose(0, 1),
    ) / math.sqrt(head_dim)
    causal_mask = positions.view(1, 1, -1) > observer_positions.view(1, -1, 1)
    logits = logits.masked_fill(causal_mask, float("-inf"))
    return torch.softmax(logits, dim=-1)


def _score_documents_by_key_value_mismatch_interaction(
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
    """Return V25 attention-weighted Key-Value mismatch interaction scores."""
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
        full_attention = _full_attention_for_kv_head(
            grouped_real_queries,
            current_key_head,
            positions,
            observer_positions,
            head_dim,
        )
        document_attention = full_attention.index_select(2, document_indices)

        current_document_keys = current_key_head.index_select(0, document_indices)
        delta_keys = current_document_keys - cached_document_keys[kv_head]
        delta_logits = torch.matmul(
            grouped_real_queries,
            delta_keys.transpose(0, 1),
        ) / math.sqrt(head_dim)
        current_document_values = current_value_heads[kv_head].index_select(
            0,
            document_indices,
        )
        value_deviation = (
            (current_document_values - cached_document_values[kv_head]).abs().sum(dim=1)
        )
        scores += (document_attention * delta_logits.abs()).sum(
            dim=(0, 1)
        ) * value_deviation

    return scores


class _KeyAwareSemanticKVShareBlender(_SemanticKVShareBlender):
    """Semantic document blender that exposes retrieved cached Keys."""

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
        """Return one key-aware score per reused document token."""
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
        """Score with untouched cached KV, then perform normal write-back."""
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

            # Current K has just received the layer's RoPE at active global
            # positions. Retrieved K follows the same post-RoPE CacheBlend
            # storage contract used by the existing V17/V19 Key comparison.
            # Score before any repair write-back so cached deltas stay intact.
            scores = self._score_documents_key_aware(
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


class LMCBlender(_KeyAwareSemanticKVShareBlender):
    """V25: attention-weighted Key-logit and Value mismatch interaction."""

    variant_name = "V25-key-value-mismatch-interaction"

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
        """Apply the V25 Key-Value mismatch interaction score."""
        return _score_documents_by_key_value_mismatch_interaction(
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
