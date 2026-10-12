# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.blender_v17 import (
    _score_documents_by_first_order_output_perturbation,
)
from lmcache.v1.compute.blend.blender_v18 import LMCBlender as V18LMCBlender

logger = init_logger(__name__)


def _score_reused_documents_by_key_l2(
    current_keys: torch.Tensor,
    cached_keys: torch.Tensor,
    document_indices: torch.Tensor,
) -> torch.Tensor:
    """Return squared Key-L2 scores for reused document tokens only.

    Args:
        current_keys: Current RoPE-applied keys for the complete prompt.
        cached_keys: Retrieved cached keys for the reusable prefix.
        document_indices: Global prompt positions of reused document tokens.

    Returns:
        One squared Key-L2 score per reused document token.
    """
    current_document_keys = current_keys.index_select(0, document_indices).to(
        torch.float32
    )
    cached_document_keys = cached_keys.index_select(0, document_indices).to(
        torch.float32
    )
    key_difference = current_document_keys - cached_document_keys
    return key_difference.flatten(start_dim=1).square().sum(dim=1)


def _select_two_stage_document_repairs(
    key_l2_scores: torch.Tensor,
    v18_scores: torch.Tensor,
    document_indices: torch.Tensor,
    mandatory_indices: torch.Tensor,
    recompute_ratio: float,
    prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Apply CacheBlend coarse filtering followed by V18 reranking.

    Args:
        key_l2_scores: One CacheBlend Key-L2 score per reused document token.
        v18_scores: One V18 perturbation score per reused document token.
        document_indices: Global prompt positions of reused document tokens.
        mandatory_indices: Global positions that must remain online-active.
        recompute_ratio: Final fraction of reused document tokens to repair.
        prompt_length: Number of tokens in the complete prompt.

    Returns:
        Sorted repaired document positions, their union with mandatory active
        positions, and the Stage-1 candidate count.

    Raises:
        ValueError: If candidates are empty, score lengths disagree, or the
            recompute ratio is outside the inclusive interval from zero to one.
    """
    document_count = document_indices.numel()
    if document_count == 0:
        raise ValueError("Cannot select repair tokens from an empty document set")
    if key_l2_scores.numel() != document_count:
        raise ValueError("Key-L2 scores and document positions must have equal length")
    if v18_scores.numel() != document_count:
        raise ValueError("V18 scores and document positions must have equal length")
    if recompute_ratio < 0.0 or recompute_ratio > 1.0:
        raise ValueError("Recompute ratio must be between zero and one")

    repair_budget = max(1, int(document_count * recompute_ratio))
    repair_budget = min(repair_budget, document_count)
    candidate_num = min(2 * repair_budget, document_count)

    # V19 uses a two-stage document repair policy.
    #
    # Stage 1: CacheBlend-style coarse filtering.
    # Rank all reused document tokens by the squared L2 difference between
    # their current and cached Key vectors, and retain 2x the final repair
    # budget as a coarse candidate set.
    #
    # Stage 2: V18 semantic perturbation reranking.
    # Compute the same Instruction+Question first-order vector-preserving
    # output-perturbation score used by V18, then rerank only the Stage-1
    # candidates and retain the final K document tokens for recomputation.
    #
    # Important:
    # - both stages operate only on reused document tokens;
    # - the V18 attention softmax is still computed over the complete
    #   causal-visible context;
    # - Stage 1 changes only the final candidate set and does not change
    #   the V18 scoring definition.

    # candidate_local_indices index the document-score vector, not the
    # original full-prompt sequence.
    candidate_local_indices = torch.topk(
        key_l2_scores,
        k=candidate_num,
    ).indices
    candidate_v18_scores = v18_scores[candidate_local_indices]
    selected_within_candidates = torch.topk(
        candidate_v18_scores,
        k=repair_budget,
    ).indices
    selected_local_indices = candidate_local_indices[selected_within_candidates]

    # Convert document-local candidate indices back to global prompt positions.
    selected_documents = document_indices[selected_local_indices]
    selected_documents, _ = torch.sort(selected_documents)
    active_positions = torch.unique(
        torch.cat((selected_documents, mandatory_indices)),
        sorted=True,
    )

    document_membership = torch.zeros(
        prompt_length,
        dtype=torch.bool,
        device=document_indices.device,
    )
    document_membership[document_indices] = True
    if not torch.all(document_membership[selected_documents]):
        raise AssertionError("Repair set contains a non-document token")
    if candidate_local_indices.numel() != candidate_num:
        raise AssertionError("Coarse candidate set does not match its budget")
    if selected_documents.numel() != repair_budget:
        raise AssertionError("Repair set does not match the final document budget")
    if not torch.all(torch.isin(mandatory_indices, active_positions)):
        raise AssertionError("Mandatory online tokens left the active set")
    return selected_documents, active_positions, candidate_num


class LMCBlender(V18LMCBlender):
    """V19: CacheBlend Key-L2 coarse filtering plus V18 reranking."""

    variant_name = "V19-cacheblend-coarse-v18-reranking"

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
        """Select document repairs with Key-L2 gating and V18 reranking."""
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

            key_l2_scores = _score_reused_documents_by_key_l2(
                k,
                old_k,
                document_indices,
            )
            # Reuse the exact V18 scorer. It computes attention over the complete
            # causal-visible context before returning one score per reused
            # document token; the Stage-1 gate is applied only afterward.
            v18_scores = _score_documents_by_first_order_output_perturbation(
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
            selected_documents, active_positions, candidate_num = (
                _select_two_stage_document_repairs(
                    key_l2_scores,
                    v18_scores,
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
                "V19 selected %d/%d reused document tokens after CacheBlend "
                "coarse filtering %d/%d candidates; active=%d",
                repair_budget,
                document_indices.numel(),
                candidate_num,
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
