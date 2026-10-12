# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

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


def _score_tokens_by_windowed_deviation_impact(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    old_values: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    observation_window_size: int,
) -> torch.Tensor:
    """Score tokens with max-reduced GQA/MQA attention and V deviation.

    This is the V11 OOM-safe approximation of:
        A = softmax(QK^T / sqrt(d_k))
        alpha = ColSum(A)
        score_i = alpha_i * ||V_i - old_V_i||_1

    Only the final ``observation_window_size`` query positions are observed.
    For GQA/MQA, V11 reduces query heads belonging to the same KV-head group
    with an elementwise maximum instead of V10's elementwise mean.
    """
    seq_len = queries.shape[0]
    group_size = num_heads // num_kv_heads
    window_size = min(max(int(observation_window_size), 1), seq_len)
    window_start = seq_len - window_size

    query_heads = queries.view(seq_len, num_heads, head_dim).to(torch.float32)
    query_groups = query_heads.view(
        seq_len, num_kv_heads, group_size, head_dim
    ).max(dim=2).values
    query_groups = query_groups[window_start:].transpose(0, 1)

    key_heads = (
        keys.view(seq_len, num_kv_heads, head_dim).transpose(0, 1).to(torch.float32)
    )
    value_heads = (
        values.view(seq_len, num_kv_heads, head_dim)
        .transpose(0, 1)
        .to(torch.float32)
    )
    old_value_heads = (
        old_values.view(seq_len, num_kv_heads, head_dim)
        .transpose(0, 1)
        .to(torch.float32)
    )

    logits = torch.matmul(query_groups, key_heads.transpose(1, 2)) / (head_dim**0.5)
    key_positions = torch.arange(seq_len, device=queries.device).view(1, 1, seq_len)
    query_positions = torch.arange(
        window_start, seq_len, device=queries.device
    ).view(1, window_size, 1)
    logits = logits.masked_fill(key_positions > query_positions, float("-inf"))

    attention_weights = torch.softmax(logits, dim=-1)
    cumulative_attention = attention_weights.sum(dim=(0, 1))
    value_deviation = (value_heads - old_value_heads).abs().sum(dim=(0, 2))
    return cumulative_attention * value_deviation


class LMCBlender:
    """V11: V10-compatible selection with max GQA/MQA head reduction."""

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
        self.num_layers = len(vllm_model.model.layers)

        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=config.blend_check_layers or [],
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
        )
        self.pyramid_plan = PyramidRecomputePlan.from_config(config, self.num_layers)
        self.observation_window_size = 64
        if config.extra_config is not None:
            self.observation_window_size = config.extra_config.get(
                "observation_window_size", self.observation_window_size
            )
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
        """Select and recompute tokens for one transformer layer."""
        logger.debug(f"Blender is processing KV for layer {layer_id}")
        old_k, old_v = self.gpu_connector.get_kv(layer_id)

        if attn_output is None:
            attn_output = torch.empty(q.shape, dtype=q.dtype, device=q.device)

        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        q, k = layer.self_attn.rotary_emb(self.metadata.positions, q, k)

        is_pyramid_layer = self.pyramid_plan.includes(layer_id)
        active_positions = self.metadata.positions
        assert active_positions is not None
        active_old_v = old_v[active_positions] if is_pyramid_layer else old_v

        if is_pyramid_layer or layer_id in self.common_metadata.check_layers:
            attn_layer = self.layerwise_model.vllm_attn_layers[layer_id]
            total_len = k.shape[0]
            assert self.common_metadata.recomp_ratios is not None
            scores = _score_tokens_by_windowed_deviation_impact(
                q,
                k,
                v,
                active_old_v,
                attn_layer.num_heads,
                attn_layer.num_kv_heads,
                attn_layer.head_size,
                self.observation_window_size,
            )

            if is_pyramid_layer:
                top_indices, original_indices = self.pyramid_plan.select(
                    scores, active_positions, old_k.shape[0], layer_id
                )
            else:
                topk_num = int(total_len * self.common_metadata.recomp_ratios[0])
                topk_num = min(max(topk_num, 1), total_len)
                top_indices = torch.topk(scores, k=topk_num).indices
                top_indices, _ = torch.sort(top_indices)
                original_indices = active_positions[top_indices]

            q, k, v, residual = (
                q[top_indices],
                k[top_indices],
                v[top_indices],
                residual[top_indices],
            )
            self.metadata.imp_indices = original_indices
            self.metadata.positions = original_indices
            attn_output = attn_output[: len(top_indices)]
            attn_metadata.update_from_top_indices(original_indices)
            logger.debug(f"Number of indices picked: {len(top_indices)}")

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
        **kwargs,
    ):
        """Perform layerwise retrieval and V11 blending."""
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
        yield

    def blend(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Perform V11 blending for the supplied tokens."""
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).cuda()

        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)
        for _ in range(self.num_layers + 2):
            next(layerwise_blender)
