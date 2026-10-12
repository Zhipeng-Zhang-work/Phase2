# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.models.utils import infer_model_from_vllm
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)


def _score_tokens_by_key_diff(
    keys: torch.Tensor,
    old_keys: torch.Tensor,
) -> torch.Tensor:
    """Score tokens by L2 distance of key changes."""
    return torch.sum(
        (keys.to(torch.float32) - old_keys.to(torch.float32)) ** 2,
        dim=[1],
    )


def _build_chunk_boundary_mask(
    token_ids: torch.Tensor,
    sep_token_ids: list[int],
    chunk_size: int,
    boundary_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build V9 boundaries from the same segments used by CacheBlend.

    RAG precompute stores the system prompt and each document prompt separately.
    RAG serving rebuilds the full prompt by joining those segments with
    ``blend_special_str``. CacheBlend uses that separator to recover segment
    ranges. V9 follows the same split, then applies EPIC-style first/last token
    retention inside each segment-local cache chunk.
    """
    seq_len = token_ids.numel()
    boundary_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    segment_ranges = _split_segment_ranges_by_separator(token_ids, sep_token_ids)

    for segment_start, segment_end in segment_ranges:
        for chunk_start in range(segment_start, segment_end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, segment_end)
            prefix_end = min(chunk_start + boundary_size, chunk_end)
            suffix_start = max(chunk_start, chunk_end - boundary_size)

            boundary_mask[chunk_start:prefix_end] = True
            boundary_mask[suffix_start:chunk_end] = True

    return boundary_mask


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


def _select_epic_l2_indices(
    scores: torch.Tensor,
    token_ids: torch.Tensor,
    sep_token_ids: list[int],
    budget: int,
    chunk_size: int,
    boundary_size: int,
) -> torch.Tensor:
    """Select V9 recompute tokens with mandatory EPIC-style boundaries.

    V9 always recomputes the first/last ``boundary_size`` tokens of every
    segment-local cache chunk. The remaining global recompute budget is filled
    by the highest-L2 non-boundary tokens. If mandatory boundary tokens already
    exceed the nominal budget, V9 keeps all boundary tokens and exceeds budget.
    """
    boundary_mask = _build_chunk_boundary_mask(
        token_ids,
        sep_token_ids,
        chunk_size,
        boundary_size,
        scores.device,
    )
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

    return torch.sort(torch.cat(selected_indices)).values


class LMCBlender:
    """
    V9: mandatory EPIC-style chunk boundaries plus L2 token selection.
    """

    def __init__(
        self,
        cache_engine,
        gpu_connector,
        vllm_model,
        config: LMCacheEngineConfig,
    ):
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector

        enable_sparse = False
        if config.extra_config is not None:
            enable_sparse = config.extra_config.get("enable_sparse", False)

        self.layerwise_model = infer_model_from_vllm(vllm_model, self, enable_sparse)

        # TODO: remove this hardcode
        self.num_layers = len(vllm_model.model.layers)

        # TODO(Jiayi): support threshold-based blending
        # TODO(Jiayi): support different ratios for different layers
        # TODO(Jiayi): support "skipping blending if hit too short"
        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=config.blend_check_layers or [],
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
        )

        extra_config = config.extra_config or {}
        default_recompute_ratio = (
            config.blend_recompute_ratios[0]
            if config.blend_recompute_ratios
            else 0.16
        )
        self.epic_recompute_ratio = float(
            extra_config.get("epic_recompute_ratio", default_recompute_ratio)
        )
        self.epic_chunk_size = int(
            extra_config.get("epic_chunk_size", getattr(config, "chunk_size", 256))
        )
        self.epic_boundary_size = int(extra_config.get("epic_boundary_size", 4))
        self.sep_token_ids = self._load_sep_token_ids()
        self._active_tokens: Optional[torch.Tensor] = None

        # This will be set during the blending process
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
    ):
        logger.debug(f"Blender is processing KV for layer {layer_id}")
        old_k, old_v = self.gpu_connector.get_kv(layer_id)

        if attn_output is None:
            attn_output = torch.empty(
                q.shape,
                dtype=q.dtype,
                device=q.device,
            )

        # perform positional encoding
        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        attn_layer = layer.self_attn
        q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)

        if layer_id in self.common_metadata.check_layers:
            key_diff_scores = _score_tokens_by_key_diff(k, old_k)
            total_len = key_diff_scores.shape[0]
            topk_num = int(total_len * self.epic_recompute_ratio)
            topk_num = min(max(topk_num, 1), total_len)
            token_ids = self._get_active_tokens(total_len)

            top_indices = _select_epic_l2_indices(
                key_diff_scores,
                token_ids,
                self.sep_token_ids,
                topk_num,
                self.epic_chunk_size,
                self.epic_boundary_size,
            )

            k, v = k[top_indices], v[top_indices]
            q = q[top_indices]
            residual = residual[top_indices]

            logger.debug(f"Number of indices picked: {len(top_indices)}")

            self.metadata.imp_indices = top_indices
            self.metadata.positions = self.metadata.positions[top_indices]
            attn_output = attn_output[: len(top_indices)]

            attn_metadata.update_from_top_indices(top_indices)

        if self.metadata.imp_indices is not None:
            old_k[self.metadata.imp_indices] = k
            old_v[self.metadata.imp_indices] = v
            return q, old_k, old_v, residual, attn_output, attn_metadata
        else:
            return q, k, v, residual, attn_output, attn_metadata

    # NOTE(Jiayi): Exposing this `blend_layer` interface as we might
    # want to ochestrate the blending process elsewhere
    def blend_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform layerwiese retrieve + blending.
        """

        # TODO(Jiayi): store is currently not included in this function

        self._active_tokens = self._to_cpu_token_tensor(tokens)
        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
        layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

        next(layerwise_retriever)
        yield

        for i in range(self.num_layers):
            next(layerwise_retriever)
            next(layerwise_model_executor)
            yield

        next(layerwise_retriever)

        self.metadata.clean()
        self._active_tokens = None
        yield

    def blend(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform blending for the given tokens.
        """

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).cuda()

        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)

        for i in range(self.num_layers + 2):
            next(layerwise_blender)

    def _load_sep_token_ids(self) -> list[int]:
        token_database = getattr(self.cache_engine, "token_database", None)
        sep_tokens = getattr(token_database, "sep_tokens", None)
        if sep_tokens is None:
            logger.warning(
                "V9 cannot read separator tokens from CacheBlend token database; "
                "falling back to one segment for boundary selection."
            )
            return []

        if isinstance(sep_tokens, torch.Tensor):
            return sep_tokens.to(device="cpu", dtype=torch.long).tolist()
        return list(sep_tokens)

    def _get_active_tokens(self, total_len: int) -> torch.Tensor:
        if self._active_tokens is None:
            return torch.arange(total_len, dtype=torch.long)
        return self._active_tokens[:total_len]

    def _to_cpu_token_tensor(
        self,
        tokens: Union[torch.Tensor, list[int]],
    ) -> torch.Tensor:
        if isinstance(tokens, torch.Tensor):
            return tokens.detach().to(device="cpu", dtype=torch.long)
        return torch.tensor(tokens, dtype=torch.long)
