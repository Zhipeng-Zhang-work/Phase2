# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Generator
from typing import Any, Optional, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.models.utils import infer_model_from_vllm
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)


def compute_key_deviation(
    recompute_keys: torch.Tensor,
    reuse_keys: torch.Tensor,
) -> torch.Tensor:
    """Compute one squared L2 key-deviation score per token.

    Args:
        recompute_keys: Newly computed keys with token dimension first.
        reuse_keys: Retrieved keys with the same shape as ``recompute_keys``.

    Returns:
        A one-dimensional tensor containing ``||K_recompute - K_reuse||_2^2``
        for every token.

    Raises:
        ValueError: If the two key tensors do not have identical shapes or are
            not at least two-dimensional.
    """
    if recompute_keys.shape != reuse_keys.shape:
        raise ValueError(
            "Recomputed and reused keys must have identical shapes, got "
            f"{tuple(recompute_keys.shape)} and {tuple(reuse_keys.shape)}"
        )
    if recompute_keys.ndim < 2:
        raise ValueError(
            "Key tensors must have a token dimension and feature dimension"
        )
    difference = (
        recompute_keys.to(torch.float32) - reuse_keys.to(torch.float32)
    ).reshape(recompute_keys.shape[0], -1)
    return torch.sum(difference * difference, dim=1)


def compute_fusion_alpha(
    recompute_keys: torch.Tensor,
    reuse_keys: torch.Tensor,
    epsilon: float = 1e-6,
    clip: bool = False,
) -> torch.Tensor:
    """Compute CacheSlide's deviation-aware fusion weight per token.

    The weight is ``||K_recompute - K_reuse||_2 / ||K_reuse||_2``.  A zero
    reused-key norm receives weight one, which safely falls back to the newly
    computed key.  Optional clipping is provided for numerical experiments;
    it is disabled by default so the paper's formula is preserved.

    Args:
        recompute_keys: Newly computed keys with token dimension first.
        reuse_keys: Retrieved keys with the same shape as ``recompute_keys``.
        epsilon: Positive lower bound used for the reused-key norm.
        clip: Whether to clip weights to ``[0, 1]``.

    Returns:
        A one-dimensional float32 tensor containing one weight per token.

    Raises:
        ValueError: If ``epsilon`` is not positive or the key shapes differ.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if recompute_keys.shape != reuse_keys.shape:
        raise ValueError(
            "Recomputed and reused keys must have identical shapes, got "
            f"{tuple(recompute_keys.shape)} and {tuple(reuse_keys.shape)}"
        )

    recompute_flat = recompute_keys.to(torch.float32).reshape(
        recompute_keys.shape[0], -1
    )
    reuse_flat = reuse_keys.to(torch.float32).reshape(reuse_keys.shape[0], -1)
    difference_norm = torch.linalg.vector_norm(recompute_flat - reuse_flat, dim=1)
    reuse_norm = torch.linalg.vector_norm(reuse_flat, dim=1)
    alpha = torch.where(
        reuse_norm > epsilon,
        difference_norm / reuse_norm.clamp_min(epsilon),
        torch.ones_like(difference_norm),
    )
    if clip:
        alpha = alpha.clamp_(0.0, 1.0)
    return alpha


def weighted_fuse(
    recompute_values: torch.Tensor,
    reuse_values: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Fuse recomputed and reused per-token values with scalar weights.

    Args:
        recompute_values: Newly computed values with token dimension first.
        reuse_values: Retrieved values with the same shape.
        alpha: One scalar fusion weight per token.

    Returns:
        A tensor with the input dtype and shape containing the weighted fusion.

    Raises:
        ValueError: If the value shapes or the number of weights disagree.
    """
    if recompute_values.shape != reuse_values.shape:
        raise ValueError(
            "Recomputed and reused values must have identical shapes, got "
            f"{tuple(recompute_values.shape)} and {tuple(reuse_values.shape)}"
        )
    if alpha.ndim != 1 or alpha.shape[0] != recompute_values.shape[0]:
        raise ValueError(
            "alpha must be one-dimensional with one value per token, got "
            f"shape {tuple(alpha.shape)} for {recompute_values.shape[0]} tokens"
        )
    alpha_view = alpha.to(dtype=recompute_values.dtype).reshape(
        alpha.shape[0], *([1] * (recompute_values.ndim - 1))
    )
    fused = alpha_view * recompute_values + (1.0 - alpha_view) * reuse_values
    return fused.to(dtype=recompute_values.dtype)


class CacheSlideWeightedBlender:
    """Fixed-Top-K CacheSlide WCA adapter for the LMCache layerwise path.

    This class deliberately implements only the weighted-fusion portion of
    CacheSlide.  The selected token set is initialized once from key deviation
    and remains fixed for the rest of the request.  Dynamic CKSim gating and
    late token promotion are not used because the current layerwise executor
    discards hidden states for unselected tokens.
    """

    fuse_values: bool = False
    variant_name: str = "CacheSlide-WCA-K"

    def __init__(
        self,
        cache_engine: Any,
        gpu_connector: Any,
        vllm_model: Any,
        config: LMCacheEngineConfig,
    ) -> None:
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector

        extra_config = config.extra_config or {}
        enable_sparse = bool(extra_config.get("enable_sparse", False))
        self.layerwise_model = infer_model_from_vllm(
            vllm_model, self, enable_sparse
        )
        self.num_layers = len(vllm_model.model.layers)

        check_layers = config.blend_check_layers or []
        if len(check_layers) != 1:
            raise ValueError(
                f"{self.variant_name} requires exactly one blend_check_layers entry"
            )
        if config.blend_pyramid_enabled:
            raise ValueError(
                f"{self.variant_name} does not support pyramid blending; "
                "set blend_pyramid_enabled=False"
            )
        if (
            config.blend_recompute_ratios is None
            or len(config.blend_recompute_ratios) != 1
        ):
            raise ValueError(
                f"{self.variant_name} requires exactly one blend_recompute_ratios entry"
            )
        self.selection_layer = check_layers[0]
        if not 0 <= self.selection_layer < self.num_layers:
            raise ValueError(
                "blend_check_layers must contain a valid layer in "
                f"[0, {self.num_layers}), "
                f"got {self.selection_layer}"
            )
        self.recompute_ratio = float(config.blend_recompute_ratios[0])
        if not 0.0 < self.recompute_ratio <= 1.0:
            raise ValueError(
                "blend_recompute_ratios must be in (0, 1], got "
                f"{self.recompute_ratio}"
            )
        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=check_layers,
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
        )

        self.alpha_epsilon = float(
            extra_config.get("cacheslide_alpha_epsilon", 1e-6)
        )
        self.alpha_clip = bool(extra_config.get("cacheslide_alpha_clip", False))
        if self.alpha_epsilon <= 0:
            raise ValueError(
                "cacheslide_alpha_epsilon must be positive, got "
                f"{self.alpha_epsilon}"
            )

        self.metadata = LMCBlendMetadata(
            imp_indices=None,
            attn_mask=None,
            positions=None,
        )
        self._alpha: Optional[torch.Tensor] = None
        self._selection_initialized = False

    def _active_reuse_kv(
        self,
        old_k: torch.Tensor,
        old_v: torch.Tensor,
        keys: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reused KV rows aligned with the currently computed rows."""
        if old_k.shape != old_v.shape:
            raise ValueError(
                "Retrieved K and V must have identical shapes, got "
                f"{tuple(old_k.shape)} and {tuple(old_v.shape)}"
            )
        if old_k.shape[0] == keys.shape[0]:
            return old_k, old_v
        if positions.ndim != 1 or positions.shape[0] != keys.shape[0]:
            raise ValueError(
                "Cannot align retrieved KV with computed rows: positions shape "
                f"{tuple(positions.shape)}, computed rows {keys.shape[0]}"
            )
        return old_k[positions], old_v[positions]

    def _select_initial_tokens(
        self,
        deviation_scores: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Select fixed Top-K indices from the positive-deviation candidates."""
        candidate_indices = torch.nonzero(
            deviation_scores > 0, as_tuple=False
        ).flatten()
        if candidate_indices.numel() == 0:
            return None

        target_count = max(
            1,
            int(deviation_scores.shape[0] * self.recompute_ratio),
        )
        target_count = min(target_count, candidate_indices.numel())
        candidate_scores = deviation_scores[candidate_indices]
        selected = candidate_indices[
            torch.topk(candidate_scores, k=target_count, sorted=False).indices
        ]
        return torch.sort(selected).values

    def _write_fused_rows(
        self,
        old_k: torch.Tensor,
        old_v: torch.Tensor,
        positions: torch.Tensor,
        recompute_k: torch.Tensor,
        recompute_v: torch.Tensor,
        reuse_k: torch.Tensor,
        reuse_v: torch.Tensor,
        alpha: torch.Tensor,
    ) -> None:
        """Write this variant's fused selected rows into the retrieved cache."""
        fused_k = weighted_fuse(recompute_k, reuse_k, alpha)
        if self.fuse_values:
            fused_v = weighted_fuse(recompute_v, reuse_v, alpha)
        else:
            fused_v = recompute_v
        old_k[positions] = fused_k
        old_v[positions] = fused_v

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
        """Select fixed Top-K rows and apply this variant's KV fusion."""
        logger.debug(
            "%s is processing KV for layer %d", self.variant_name, layer_id
        )
        old_k, old_v = self.gpu_connector.get_kv(layer_id)

        if attn_output is None:
            attn_output = torch.empty(q.shape, dtype=q.dtype, device=q.device)

        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )
        positions = self.metadata.positions
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        q, k = layer.self_attn.rotary_emb(positions, q, k)
        reuse_k, reuse_v = self._active_reuse_kv(old_k, old_v, k, positions)

        if (
            layer_id == self.selection_layer
            and not self._selection_initialized
        ):
            deviation_scores = compute_key_deviation(k, reuse_k)
            top_indices = self._select_initial_tokens(deviation_scores)
            self._selection_initialized = True
            if top_indices is None:
                logger.info(
                    "%s found no positive key deviations at layer %d; "
                    "keeping the retrieved KV cache",
                    self.variant_name,
                    layer_id,
                )
                self._alpha = None
            else:
                selected_positions = positions[top_indices]
                selected_recompute_k = k[top_indices]
                selected_recompute_v = v[top_indices]
                selected_reuse_k = reuse_k[top_indices]
                selected_reuse_v = reuse_v[top_indices]
                self._alpha = compute_fusion_alpha(
                    selected_recompute_k,
                    selected_reuse_k,
                    epsilon=self.alpha_epsilon,
                    clip=self.alpha_clip,
                )

                # The initialization layer is the full-recompute comparison
                # layer. Weighted fusion starts on the following layer, as in
                # CacheSlide's WCA pseudocode.
                old_k[selected_positions] = selected_recompute_k
                old_v[selected_positions] = selected_recompute_v
                q = q[top_indices]
                k = selected_recompute_k
                v = selected_recompute_v
                residual = residual[top_indices]
                self.metadata.imp_indices = selected_positions
                self.metadata.positions = selected_positions
                attn_output = attn_output[: len(top_indices)]
                attn_metadata.update_from_top_indices(selected_positions)
                logger.info(
                    "%s selected %d/%d tokens (ratio=%.6f) at layer %d; "
                    "alpha[min=%.6f max=%.6f mean=%.6f]",
                    self.variant_name,
                    len(selected_positions),
                    len(positions),
                    len(selected_positions) / len(positions),
                    layer_id,
                    self._alpha.min().item(),
                    self._alpha.max().item(),
                    self._alpha.mean().item(),
                )
                return (
                    q,
                    old_k,
                    old_v,
                    residual,
                    attn_output,
                    attn_metadata,
                )

        if self.metadata.imp_indices is not None:
            selected_positions = self.metadata.imp_indices
            selected_reuse_k = old_k[selected_positions]
            selected_reuse_v = old_v[selected_positions]
            if self._alpha is None:
                self._alpha = compute_fusion_alpha(
                    k,
                    selected_reuse_k,
                    epsilon=self.alpha_epsilon,
                    clip=self.alpha_clip,
                )
            alpha = self._alpha
            self._write_fused_rows(
                old_k,
                old_v,
                selected_positions,
                k,
                v,
                selected_reuse_k,
                selected_reuse_v,
                alpha,
            )
            # Update alpha for the next layer using the unfused current-layer
            # K and the cache value that entered this layer.
            self._alpha = compute_fusion_alpha(
                k,
                selected_reuse_k,
                epsilon=self.alpha_epsilon,
                clip=self.alpha_clip,
            )
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
        """Perform layerwise retrieval and fixed-Top-K WCA blending."""
        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
        layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

        next(layerwise_retriever)
        yield

        for _ in range(self.num_layers):
            next(layerwise_retriever)
            next(layerwise_model_executor)
            yield

        next(layerwise_retriever)
        self._alpha = None
        self._selection_initialized = False
        self.metadata.clean()
        yield

    def blend(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        """Perform fixed-Top-K WCA blending for the supplied tokens."""
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).cuda()

        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)
        for _ in range(self.num_layers + 2):
            next(layerwise_blender)
