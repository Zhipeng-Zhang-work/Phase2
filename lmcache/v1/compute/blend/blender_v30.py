# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Generator
from typing import Any, Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.blend.pyramid import PyramidRecomputePlan
from lmcache.v1.compute.models.utils import infer_model_from_vllm
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)


def _score_tokens_by_joint_kv_diff(
    keys: torch.Tensor,
    old_keys: torch.Tensor,
    values: torch.Tensor,
    old_values: torch.Tensor,
) -> torch.Tensor:
    """Return V2 joint mismatch for aligned token rows, entirely in float32.

    Args:
        keys: Current RoPE-applied keys, shaped [tokens, flattened KV width].
        old_keys: Retrieved keys with the same token-row layout.
        values: Current values with the same token-row layout.
        old_values: Retrieved values with the same token-row layout.

    Returns:
        One score per token: squared Key-L2 times squared Value-L2,
        without normalization or per-head aggregation.
    """
    key_diff_scores = torch.sum(
        (keys.to(torch.float32) - old_keys.to(torch.float32)) ** 2,
        dim=[1],
    )
    value_diff_scores = torch.sum(
        (values.to(torch.float32) - old_values.to(torch.float32)) ** 2,
        dim=[1],
    )
    return key_diff_scores * value_diff_scores


def _split_segment_ranges_by_separator(
    token_ids: torch.Tensor,
    sep_token_ids: list[int],
) -> list[tuple[int, int]]:
    """Split token positions by CacheBlend's separator token sequence."""
    total_len = token_ids.numel()
    if not sep_token_ids or total_len == 0:
        return [(0, total_len)]

    tokens = token_ids.cpu().tolist()
    sep_len = len(sep_token_ids)
    ranges: list[tuple[int, int]] = []
    start = 0
    idx = 0

    while idx <= total_len - sep_len:
        if tokens[idx : idx + sep_len] == sep_token_ids:
            if start < idx:
                ranges.append((start, idx))
            start = idx + sep_len
            idx = start
        else:
            idx += 1

    if start < total_len:
        ranges.append((start, total_len))

    return ranges


def _build_precompute_segment_boundary_mask(
    token_ids: torch.Tensor,
    sep_token_ids: list[int],
    boundary_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Build mandatory boundaries for independently precomputed documents.

    Args:
        token_ids: Original reusable-prefix token IDs.
        sep_token_ids: CacheBlend separator token sequence.
        boundary_size: Number of tokens retained at each document edge.
        device: Device for the returned mask.

    Returns:
        The full-prefix boundary mask and recovered document-segment count.
    """
    sequence_length = token_ids.numel()
    boundary_mask = torch.zeros(
        sequence_length,
        dtype=torch.bool,
        device=device,
    )
    if not sep_token_ids or sequence_length == 0:
        return boundary_mask, 0

    segment_ranges = _split_segment_ranges_by_separator(token_ids, sep_token_ids)
    # The first separator-delimited segment is the system prompt. Every
    # following range is one independently precomputed document/context.
    document_segment_ranges = segment_ranges[1:]

    # V30 treats each independently precomputed document/context as one
    # semantic repair segment.
    #
    # The CacheBlend separator reconstructs the same segment boundaries
    # that were used during offline document-level KV precomputation.
    #
    # IMPORTANT:
    # LMCache's chunk_size (e.g. 256 tokens) is only the cache
    # storage/matching granularity. It is NOT a semantic document boundary
    # and must not be used to subdivide a precomputed ctx/document here.
    #
    # For every independently precomputed document segment, V30
    # mandatorily repairs the first and last boundary_size tokens.
    # The remaining global repair budget is filled with V2 joint
    # squared Key-L2 times squared Value-L2 mismatch scores.
    for segment_start, segment_end in document_segment_ranges:
        # Each returned document range corresponds to one independently
        # precomputed ctx/document, not to an LMCache storage chunk.
        prefix_end = min(segment_start + boundary_size, segment_end)
        suffix_start = max(segment_start, segment_end - boundary_size)
        boundary_mask[segment_start:prefix_end] = True
        boundary_mask[suffix_start:segment_end] = True

    return boundary_mask, len(document_segment_ranges)


def _select_segment_boundary_joint_kv_indices(
    scores: torch.Tensor,
    token_ids: torch.Tensor,
    active_positions: torch.Tensor,
    sep_token_ids: list[int],
    budget: int,
    boundary_size: int,
) -> tuple[torch.Tensor, int, int]:
    """Keep document boundaries and fill the remaining budget by joint K/V mismatch.

    Args:
        scores: Joint K/V mismatch scores for currently participating tokens.
        token_ids: Original reusable-prefix token IDs.
        active_positions: Global positions corresponding to ``scores``.
        sep_token_ids: CacheBlend separator token sequence.
        budget: Nominal global recomputation budget.
        boundary_size: Number of mandatory tokens at each document edge.

    Returns:
        Sorted local selected indices, mandatory boundary-token count, and
        independently precomputed document-segment count.
    """
    full_boundary_mask, document_segment_count = (
        _build_precompute_segment_boundary_mask(
            token_ids,
            sep_token_ids,
            boundary_size,
            scores.device,
        )
    )
    boundary_mask = full_boundary_mask[active_positions]
    boundary_indices = torch.nonzero(boundary_mask, as_tuple=False).flatten()

    selected_indices = [boundary_indices]
    remaining_budget = max(budget - boundary_indices.numel(), 0)
    non_boundary_indices = torch.nonzero(~boundary_mask, as_tuple=False).flatten()

    if remaining_budget > 0 and non_boundary_indices.numel() > 0:
        remaining_budget = min(remaining_budget, non_boundary_indices.numel())
        selected_positions = torch.topk(
            scores[non_boundary_indices],
            k=remaining_budget,
        ).indices
        selected_indices.append(non_boundary_indices[selected_positions])

    selected = torch.sort(torch.cat(selected_indices)).values
    if not torch.all(torch.isin(boundary_indices, selected)):
        raise AssertionError("A mandatory document boundary token was dropped")
    if selected.numel() < boundary_indices.numel():
        raise AssertionError("Selected set is smaller than its mandatory boundaries")
    return selected, boundary_indices.numel(), document_segment_count


class LMCBlender:
    """V30: semantic document boundaries (4+4) plus joint Key/Value mismatch."""

    def __init__(
        self,
        cache_engine: Any,
        gpu_connector: Any,
        vllm_model: Any,
        config: LMCacheEngineConfig,
    ) -> None:
        """Initialize V30 from the standard CacheBlend model and config."""
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector

        enable_sparse = False
        if config.extra_config is not None:
            enable_sparse = config.extra_config.get("enable_sparse", False)

        self.layerwise_model = infer_model_from_vllm(
            vllm_model,
            self,
            enable_sparse,
        )
        self.num_layers = len(vllm_model.model.layers)
        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=config.blend_check_layers or [],
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
        )
        self.pyramid_plan = PyramidRecomputePlan.from_config(
            config,
            self.num_layers,
        )

        extra_config = config.extra_config or {}
        default_recompute_ratio = (
            config.blend_recompute_ratios[0] if config.blend_recompute_ratios else 0.16
        )
        self.recompute_ratio = float(
            extra_config.get("epic_recompute_ratio", default_recompute_ratio)
        )
        # Fixed ablation setting; external boundary overrides do not apply.
        self.boundary_size = 4
        if self.boundary_size < 0:
            raise ValueError("V30 boundary size must be non-negative")
        self.sep_token_ids = self._load_sep_token_ids()
        self._active_tokens: Optional[torch.Tensor] = None

        self.metadata = LMCBlendMetadata(
            imp_indices=None,
            attn_mask=None,
            positions=None,
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
        """Select mandatory document edges and joint K/V mismatch repair tokens."""
        logger.debug("V30 is processing KV for layer %d", layer_id)
        old_k, old_v = self.gpu_connector.get_kv(layer_id)

        if attn_output is None:
            attn_output = torch.empty(q.shape, dtype=q.dtype, device=q.device)

        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0],
                device=q.device,
                dtype=torch.int64,
            )
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        q, k = layer.self_attn.rotary_emb(self.metadata.positions, q, k)

        is_pyramid_layer = self.pyramid_plan.includes(layer_id)
        active_positions = self.metadata.positions
        assert active_positions is not None
        active_old_k = old_k[active_positions] if is_pyramid_layer else old_k
        active_old_v = old_v[active_positions] if is_pyramid_layer else old_v

        if is_pyramid_layer or layer_id in self.common_metadata.check_layers:
            joint_kv_scores = _score_tokens_by_joint_kv_diff(
                k, active_old_k, v, active_old_v
            )
            participating_count = joint_kv_scores.shape[0]
            if is_pyramid_layer:
                nominal_budget = self.pyramid_plan.target_count(
                    old_k.shape[0],
                    participating_count,
                    layer_id,
                )
            else:
                nominal_budget = int(participating_count * self.recompute_ratio)
                nominal_budget = min(max(nominal_budget, 1), participating_count)
            token_ids = self._get_original_tokens(old_k.shape[0])

            top_indices, boundary_count, document_segment_count = (
                _select_segment_boundary_joint_kv_indices(
                    joint_kv_scores,
                    token_ids,
                    active_positions,
                    self.sep_token_ids,
                    nominal_budget,
                    self.boundary_size,
                )
            )

            k, v = k[top_indices], v[top_indices]
            q = q[top_indices]
            residual = residual[top_indices]

            original_indices = active_positions[top_indices]
            self.metadata.imp_indices = original_indices
            self.metadata.positions = original_indices
            attn_output = attn_output[: top_indices.numel()]
            attn_metadata.update_from_top_indices(original_indices)

            logger.info(
                "V30 document_segment_count=%d; boundary_size=%d; "
                "mandatory_boundary_tokens=%d; nominal_budget=%d; "
                "remaining_joint_kv_budget=%d; final_repaired_tokens=%d",
                document_segment_count,
                self.boundary_size,
                boundary_count,
                nominal_budget,
                max(nominal_budget - boundary_count, 0),
                top_indices.numel(),
            )
            if boundary_count > nominal_budget:
                logger.warning(
                    "V30 mandatory_boundary_tokens=%d exceeds nominal_budget=%d; "
                    "final_selected_count=%d; all mandatory boundaries retained",
                    boundary_count,
                    nominal_budget,
                    top_indices.numel(),
                )

        if self.metadata.imp_indices is not None:
            old_k[self.metadata.imp_indices] = k
            old_v[self.metadata.imp_indices] = v
            return q, old_k, old_v, residual, attn_output, attn_metadata

        if old_k.shape == k.shape and old_v.shape == v.shape:
            old_k.copy_(k)
            old_v.copy_(v)
            return q, old_k, old_v, residual, attn_output, attn_metadata

        return q, k, v, residual, attn_output, attn_metadata

    def blend_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Generator[None, None, None]:
        """Run layerwise retrieval and V30 selection for one token sequence."""
        self._active_tokens = self._to_cpu_token_tensor(tokens)
        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
        layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

        next(layerwise_retriever)
        yield

        for _ in range(self.num_layers):
            next(layerwise_retriever)
            next(layerwise_model_executor)
            yield

        next(layerwise_retriever)
        self.metadata.clean()
        self._active_tokens = None
        yield

    def blend(
        self,
        tokens: torch.Tensor | list[int],
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        """Perform V30 blending for one reusable token sequence."""
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).cuda()

        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)
        for _ in range(self.num_layers + 2):
            next(layerwise_blender)

    def _load_sep_token_ids(self) -> list[int]:
        token_database = getattr(self.cache_engine, "token_database", None)
        sep_tokens = getattr(token_database, "sep_tokens", None)
        if sep_tokens is None:
            logger.warning(
                "V30 cannot read separator tokens from the CacheBlend token "
                "database; no document boundary prior will be applied."
            )
            return []

        if isinstance(sep_tokens, torch.Tensor):
            return sep_tokens.to(device="cpu", dtype=torch.long).tolist()
        return list(sep_tokens)

    def _get_original_tokens(self, total_len: int) -> torch.Tensor:
        if self._active_tokens is None:
            return torch.arange(total_len, dtype=torch.long)
        return self._active_tokens[:total_len]

    def _to_cpu_token_tensor(
        self,
        tokens: torch.Tensor | list[int],
    ) -> torch.Tensor:
        if isinstance(tokens, torch.Tensor):
            return tokens.detach().to(device="cpu", dtype=torch.long)
        return torch.tensor(tokens, dtype=torch.long)
