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


def _score_tokens_by_attention_perturbation(
    queries: torch.Tensor,
    keys: torch.Tensor,
    old_keys: torch.Tensor,
    values: torch.Tensor,
    old_values: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Approximate token impact with a first-order attention perturbation score."""

    seq_len = queries.shape[0]
    group_size = num_heads // num_kv_heads

    query_heads = queries.view(seq_len, num_heads, head_dim).to(torch.float32)
    query_groups = query_heads.view(
        seq_len,
        num_kv_heads,
        group_size,
        head_dim,
    ).mean(dim=2)
    query_groups = query_groups.transpose(0, 1)

    key_heads = keys.view(seq_len, num_kv_heads, head_dim).transpose(0, 1).to(
        torch.float32
    )
    old_key_heads = old_keys.view(seq_len, num_kv_heads, head_dim).transpose(
        0, 1
    ).to(torch.float32)
    value_heads = values.view(seq_len, num_kv_heads, head_dim).transpose(0, 1).to(
        torch.float32
    )
    old_value_heads = old_values.view(seq_len, num_kv_heads, head_dim).transpose(
        0, 1
    ).to(torch.float32)

    delta_k = key_heads - old_key_heads
    delta_v = value_heads - old_value_heads

    logits = torch.matmul(query_groups, key_heads.transpose(1, 2)) / (head_dim**0.5)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=queries.device),
        diagonal=1,
    )
    logits = logits.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
    attention_weights = torch.softmax(logits, dim=-1)

    delta_logits = torch.matmul(query_groups, delta_k.transpose(1, 2)).abs() / (
        head_dim**0.5
    )
    key_sensitivity = (attention_weights * delta_logits).mean(dim=1)

    outputs = torch.matmul(attention_weights, value_heads)
    mean_outputs = outputs.mean(dim=1)
    value_leverage = torch.linalg.vector_norm(
        value_heads - mean_outputs.unsqueeze(1),
        dim=-1,
    )
    value_error = torch.linalg.vector_norm(delta_v, dim=-1)
    attention_mass = attention_weights.mean(dim=1)

    key_effect = key_sensitivity * value_leverage
    value_effect = attention_mass * value_error
    return (key_effect + value_effect).mean(dim=0)


class LMCBlender:
    """
    V6: query-aware attention perturbation scoring
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

        #是否启用稀疏注意力
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
            num_heads = self.layerwise_model.vllm_attn_layers[layer_id].num_heads
            num_kv_heads = self.layerwise_model.vllm_attn_layers[layer_id].num_kv_heads
            head_dim = self.layerwise_model.vllm_attn_layers[layer_id].head_size
            key_scores = _score_tokens_by_attention_perturbation(
                q,
                k,
                old_k,
                v,
                old_v,
                num_heads,
                num_kv_heads,
                head_dim,
            )
            total_len = key_scores.shape[0]

            assert self.common_metadata.recomp_ratios is not None

            # TODO(Jiayi): remove `[0]` hardcode
            topk_num = int(total_len * self.common_metadata.recomp_ratios[0])
            topk_num = max(topk_num, 1)

            top_indices = torch.topk(key_scores, k=topk_num).indices
            top_indices, _ = torch.sort(top_indices)

            k, v = k[top_indices], v[top_indices]
            q = q[top_indices]
            residual = residual[top_indices]

            logger.debug(f"Number of indices picked: {len(top_indices)}")

            self.metadata.imp_indices = top_indices
            self.metadata.positions = self.metadata.positions[top_indices]
            attn_output = attn_output[:topk_num]

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
