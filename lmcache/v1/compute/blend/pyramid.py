# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional

# Third Party
import torch

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig


class PyramidRecomputePlan:
    """Describe the per-layer token budgets for pyramid recomputation."""

    def __init__(
        self,
        enabled: bool,
        start_layer: Optional[int],
        layer_ratios: dict[int, float],
    ) -> None:
        self.enabled = enabled
        self.start_layer = start_layer
        self.layer_ratios = layer_ratios

    @classmethod
    def from_config(
        cls,
        config: "LMCacheEngineConfig",
        num_layers: int,
    ) -> "PyramidRecomputePlan":
        """Build a pyramid recompute plan from an LMCache configuration.

        Args:
            config: LMCache engine configuration containing pyramid settings.
            num_layers: Total number of transformer layers.

        Returns:
            A disabled plan when pyramid blending is off, otherwise a plan
            covering the configured start layer through the final layer.

        Raises:
            ValueError: If enabled pyramid settings do not contain exactly one
                start layer and one average recompute ratio.
        """
        if not config.blend_pyramid_enabled:
            return cls(False, None, {})

        check_layers = config.blend_check_layers or []
        recompute_ratios = config.blend_recompute_ratios or []
        if len(check_layers) != 1:
            raise ValueError(
                "Pyramid blending requires exactly one start layer in "
                "blend_check_layers"
            )
        if len(recompute_ratios) != 1:
            raise ValueError(
                "Pyramid blending requires exactly one average ratio in "
                "blend_recompute_ratios"
            )

        start_layer = check_layers[0]
        ratios = build_pyramid_recompute_ratios(
            num_layers=num_layers,
            start_layer=start_layer,
            average_ratio=recompute_ratios[0],
            beta=config.blend_pyramid_beta,
        )
        return cls(
            True,
            start_layer,
            {start_layer + offset: ratio for offset, ratio in enumerate(ratios)},
        )

    def includes(self, layer_id: int) -> bool:
        """Return whether a layer uses pyramid token selection.

        Args:
            layer_id: Zero-based transformer layer index.

        Returns:
            ``True`` when the layer has a pyramid recompute budget.
        """
        return self.enabled and layer_id in self.layer_ratios

    def select(
        self,
        scores: torch.Tensor,
        active_positions: torch.Tensor,
        original_seq_len: int,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the nested token subset configured for one layer.

        Args:
            scores: Importance score for every currently active token.
            active_positions: Original prompt position of each active token.
            original_seq_len: Prompt length before pyramid selection.
            layer_id: Zero-based transformer layer index.

        Returns:
            Local active-sequence indices and corresponding original positions.

        Raises:
            ValueError: If ``layer_id`` is not part of this plan.
        """
        if not self.includes(layer_id):
            raise ValueError(f"Layer {layer_id} is not in the pyramid plan")
        return select_pyramid_tokens(
            scores=scores,
            active_positions=active_positions,
            original_seq_len=original_seq_len,
            target_ratio=self.layer_ratios[layer_id],
        )

    def target_count(
        self,
        original_seq_len: int,
        active_token_count: int,
        layer_id: int,
    ) -> int:
        """Return a layer budget capped by the current active token count.

        Args:
            original_seq_len: Prompt length before pyramid selection.
            active_token_count: Number of tokens still active at this layer.
            layer_id: Zero-based transformer layer index.

        Returns:
            The configured token budget, with at least one active token.

        Raises:
            ValueError: If the layer is not in the plan or lengths are invalid.
        """
        if not self.includes(layer_id):
            raise ValueError(f"Layer {layer_id} is not in the pyramid plan")
        if original_seq_len <= 0:
            raise ValueError(
                f"original_seq_len must be positive, got {original_seq_len}"
            )
        if active_token_count <= 0:
            raise ValueError(
                f"active_token_count must be positive, got {active_token_count}"
            )
        target_count = round(original_seq_len * self.layer_ratios[layer_id])
        return min(max(target_count, 1), active_token_count)


def build_pyramid_recompute_ratios(
    num_layers: int,
    start_layer: int,
    average_ratio: float,
    beta: float,
) -> list[float]:
    """Build monotonically decreasing per-layer recompute ratios.

    The returned ratios cover ``start_layer`` through the final model layer.
    Their arithmetic mean equals ``average_ratio``. ``beta`` controls the
    ratio between the first and last pyramid layers; a value of one produces
    a constant schedule.

    Args:
        num_layers: Total number of transformer layers.
        start_layer: Zero-based first layer using pyramid recomputation.
        average_ratio: Average fraction of original prompt tokens recomputed.
        beta: Ratio between the first and last layer budgets. Must be >= 1.

    Returns:
        A ratio for every layer from ``start_layer`` to the final layer.

    Raises:
        ValueError: If a layer, ratio, or beta argument is invalid.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if not 0 <= start_layer < num_layers:
        raise ValueError(f"start_layer must be in [0, {num_layers}), got {start_layer}")
    if not 0.0 <= average_ratio <= 1.0:
        raise ValueError(
            "average_ratio must be between 0 and 1, " f"got {average_ratio}"
        )
    if beta < 1.0:
        raise ValueError(f"beta must be at least 1, got {beta}")

    pyramid_layer_count = num_layers - start_layer
    if pyramid_layer_count == 1:
        return [average_ratio]

    min_ratio = 2.0 * average_ratio / (beta + 1.0)
    max_ratio = 2.0 * average_ratio - min_ratio
    if max_ratio > 1.0:
        max_ratio = 1.0
        min_ratio = 2.0 * average_ratio - max_ratio

    step = (max_ratio - min_ratio) / (pyramid_layer_count - 1)
    return [
        max_ratio - layer_offset * step for layer_offset in range(pyramid_layer_count)
    ]


def select_pyramid_tokens(
    scores: torch.Tensor,
    active_positions: torch.Tensor,
    original_seq_len: int,
    target_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a nested token subset while preserving original positions.

    Args:
        scores: One importance score for each currently active token.
        active_positions: Original prompt position of each active token.
        original_seq_len: Token count before the first pyramid selection.
        target_ratio: Target fraction relative to the original prompt length.

    Returns:
        A pair containing indices local to the active sequence and their
        corresponding original prompt positions. Both are sorted by sequence
        order.

    Raises:
        ValueError: If tensor lengths or target arguments are invalid.
    """
    if scores.ndim != 1:
        raise ValueError(f"scores must be one-dimensional, got {scores.shape}")
    if active_positions.ndim != 1:
        raise ValueError(
            "active_positions must be one-dimensional, " f"got {active_positions.shape}"
        )
    if len(scores) != len(active_positions):
        raise ValueError(
            "scores and active_positions must have equal lengths, "
            f"got {len(scores)} and {len(active_positions)}"
        )
    if original_seq_len <= 0:
        raise ValueError(f"original_seq_len must be positive, got {original_seq_len}")
    if not 0.0 <= target_ratio <= 1.0:
        raise ValueError(f"target_ratio must be between 0 and 1, got {target_ratio}")
    if len(scores) == 0:
        raise ValueError("Cannot select from an empty active token sequence")

    target_count = round(original_seq_len * target_ratio)
    target_count = min(max(target_count, 1), len(scores))
    if target_count == len(scores):
        local_indices = torch.arange(len(scores), device=scores.device)
    else:
        local_indices = torch.topk(scores, k=target_count).indices
        local_indices, _ = torch.sort(local_indices)

    return local_indices, active_positions[local_indices]
