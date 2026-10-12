# SPDX-License-Identifier: Apache-2.0
# Standard
import math
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


def _repeat_kv_heads(keys: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """Repeat KV heads to align with query heads under GQA."""

    return keys.repeat_interleave(num_key_value_groups, dim=0)


def _score_keys_by_recent_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    window_size: int,
    kernel_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Score tokens using recent-query attention over the full key sequence."""

    seq_len = keys.shape[0]
    window_size = min(window_size, seq_len)
    num_key_value_groups = num_heads // num_kv_heads

    query_heads = queries.view(seq_len, num_heads, head_dim).transpose(0, 1)
    key_heads = keys.view(seq_len, num_kv_heads, head_dim).transpose(0, 1)
    repeated_key_heads = _repeat_kv_heads(key_heads, num_key_value_groups)

    recent_queries = query_heads[:, -window_size:, :]
    attn_scores = torch.matmul(
        recent_queries.to(torch.float32),
        repeated_key_heads.transpose(1, 2).to(torch.float32),
    ) / math.sqrt(head_dim)

    key_positions = torch.arange(seq_len, device=keys.device)
    query_positions = torch.arange(seq_len - window_size, seq_len, device=keys.device)
    causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    attn_scores = attn_scores.masked_fill(
        causal_mask.unsqueeze(0),
        float("-inf"),
    )
    attn_weights = F.softmax(attn_scores, dim=-1)

    if seq_len > window_size:
        historical_scores = attn_weights[..., : seq_len - window_size]
        token_scores = historical_scores.mean(dim=1)
        token_scores = F.avg_pool1d(
            token_scores.unsqueeze(0),
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        ).squeeze(0)
        token_scores = token_scores.mean(dim=0)
        keep_recent_score = token_scores.max()
        return F.pad(token_scores, (0, window_size), value=keep_recent_score.item())

    return attn_weights.mean(dim=1).mean(dim=0)


class LMCBlender:
    """
    V4: snapkv scoring style
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
        self.observation_window_size = 64
        self.kernel_size = 5
        if config.extra_config is not None:
            self.observation_window_size = config.extra_config.get(
                "observation_window_size",
                self.observation_window_size,
            )
            self.kernel_size = config.extra_config.get(
                "kernel_size",
                self.kernel_size,
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

        if is_pyramid_layer or layer_id in self.common_metadata.check_layers:
            num_heads = self.layerwise_model.vllm_attn_layers[layer_id].num_heads
            num_kv_heads = self.layerwise_model.vllm_attn_layers[layer_id].num_kv_heads
            head_dim = self.layerwise_model.vllm_attn_layers[layer_id].head_size
            key_scores = _score_keys_by_recent_attention(
                q,
                k,
                self.observation_window_size,
                self.kernel_size,
                num_heads,
                num_kv_heads,
                head_dim,
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
