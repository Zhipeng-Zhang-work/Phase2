# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
import torch
import torch.nn.functional as F

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.blend.pyramid import PyramidRecomputePlan
from lmcache.v1.compute.models.utils import infer_model_from_vllm
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)


def _apply_local_approximation(
    scores: torch.Tensor,
    local_window_size: int,
) -> torch.Tensor:
    """Normalize per-head scores within local windows."""

    if local_window_size <= 1:
        return scores

    _, seq_len = scores.shape
    pad_len = (local_window_size - seq_len % local_window_size) % local_window_size
    padded_scores = F.pad(scores, (0, pad_len))
    reshaped_scores = padded_scores.reshape(scores.shape[0], -1, local_window_size)
    normalized_scores = reshaped_scores / reshaped_scores.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)
    return normalized_scores.reshape(scores.shape[0], -1)[:, :seq_len]


def _score_tokens_by_local_kv_difference(
    keys: torch.Tensor,
    old_keys: torch.Tensor,
    values: torch.Tensor,
    old_values: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    local_window_size: int,
) -> torch.Tensor:
    """Score tokens with per-KV-head local approximated K/V differences."""

    seq_len = keys.shape[0]
    key_heads = keys.view(seq_len, num_kv_heads, head_dim).transpose(0, 1)
    old_key_heads = old_keys.view(seq_len, num_kv_heads, head_dim).transpose(0, 1)
    value_heads = values.view(seq_len, num_kv_heads, head_dim).transpose(0, 1)
    old_value_heads = old_values.view(seq_len, num_kv_heads, head_dim).transpose(0, 1)

    k2 = ((key_heads.to(torch.float32) - old_key_heads.to(torch.float32)) ** 2).sum(
        dim=-1
    )
    v2 = ((value_heads.to(torch.float32) - old_value_heads.to(torch.float32)) ** 2).sum(
        dim=-1
    )

    k2 = _apply_local_approximation(k2, local_window_size)
    v2 = _apply_local_approximation(v2, local_window_size)

    return (k2 * v2).mean(dim=0)


class LMCBlender:
    """
    V5: local K/V difference scoring style, curkv
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

        # 是否启用稀疏注意力
        enable_sparse = False
        if config.extra_config is not None:
            enable_sparse = config.extra_config.get("enable_sparse", False)

        self.layerwise_model = infer_model_from_vllm(vllm_model, self, enable_sparse)

        # TODO: remove this hardcode
        self.num_layers = len(vllm_model.model.layers)

        # TODO(Jiayi): support threshold-based blending
        # TODO(Jiayi): support different ratios for different layers
        # TODO(Jiayi): support "skipping blending if hit too short"
        check_layers = config.blend_check_layers or []
        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=check_layers,
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
        )
        self.pyramid_plan = PyramidRecomputePlan.from_config(
            config,
            self.num_layers,
        )
        self.local_window_size = 16
        if config.extra_config is not None:
            self.local_window_size = config.extra_config.get(
                "local_window_size",
                self.local_window_size,
            )

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

        is_pyramid_layer = self.pyramid_plan.includes(layer_id)
        active_positions = self.metadata.positions
        assert active_positions is not None
        active_old_k = old_k[active_positions] if is_pyramid_layer else old_k
        active_old_v = old_v[active_positions] if is_pyramid_layer else old_v

        if is_pyramid_layer or layer_id in self.common_metadata.check_layers:
            num_kv_heads = self.layerwise_model.vllm_attn_layers[layer_id].num_kv_heads
            head_dim = self.layerwise_model.vllm_attn_layers[layer_id].head_size
            key_scores = _score_tokens_by_local_kv_difference(
                k,
                active_old_k,
                v,
                active_old_v,
                num_kv_heads,
                head_dim,
                self.local_window_size,
            )
            total_len = key_scores.shape[0]

            assert self.common_metadata.recomp_ratios is not None

            if is_pyramid_layer:
                top_indices, original_indices = self.pyramid_plan.select(
                    key_scores,
                    active_positions,
                    old_k.shape[0],
                    layer_id,
                )
            else:
                topk_num = int(total_len * self.common_metadata.recomp_ratios[0])
                topk_num = max(topk_num, 1)
                top_indices = torch.topk(key_scores, k=topk_num).indices
                top_indices, _ = torch.sort(top_indices)
                original_indices = active_positions[top_indices]

            k, v = k[top_indices], v[top_indices]
            q = q[top_indices]
            residual = residual[top_indices]

            logger.debug(f"Number of indices picked: {len(top_indices)}")

            self.metadata.imp_indices = original_indices
            self.metadata.positions = original_indices
            attn_output = attn_output[: len(top_indices)]

            attn_metadata.update_from_top_indices(original_indices)

        if self.metadata.imp_indices is not None:
            old_k[self.metadata.imp_indices] = k
            old_v[self.metadata.imp_indices] = v
            return q, old_k, old_v, residual, attn_output, attn_metadata

        if old_k.shape == k.shape and old_v.shape == v.shape:
            # Before the checking layer, the full prompt KV has already been
            # recomputed. Keep the cache consistent by replacing the retrieved
            # KV with the freshly computed KV for the whole layer.
            old_k.copy_(k)
            old_v.copy_(v)
            return q, old_k, old_v, residual, attn_output, attn_metadata

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
