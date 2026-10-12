# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1.compute.blend.pyramid import (
    PyramidRecomputePlan,
    build_pyramid_recompute_ratios,
    select_pyramid_tokens,
)
from lmcache.v1.config import LMCacheEngineConfig


def test_pyramid_schedule_preserves_average_ratio() -> None:
    """The layer schedule should be decreasing with the configured mean."""
    ratios = build_pyramid_recompute_ratios(
        num_layers=64,
        start_layer=1,
        average_ratio=0.16,
        beta=20,
    )

    assert len(ratios) == 63
    assert all(current >= following for current, following in zip(ratios, ratios[1:]))
    assert sum(ratios) / len(ratios) == pytest.approx(0.16)
    assert ratios[0] > 0.16
    assert ratios[-1] < 0.16


def test_pyramid_schedule_beta_one_is_constant() -> None:
    """Beta one should reduce pyramid mode to a constant token budget."""
    ratios = build_pyramid_recompute_ratios(
        num_layers=8,
        start_layer=2,
        average_ratio=0.25,
        beta=1,
    )

    assert ratios == pytest.approx([0.25] * 6)


def test_pyramid_selection_composes_original_positions() -> None:
    """Repeated selection should retain positions in the original prompt."""
    active_positions = torch.arange(10)
    first_scores = torch.tensor([0.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    _, first_positions = select_pyramid_tokens(
        scores=first_scores,
        active_positions=active_positions,
        original_seq_len=10,
        target_ratio=0.5,
    )

    second_scores = torch.tensor([0.0, 0.0, 10.0, 9.0, 8.0])
    second_local_indices, second_positions = select_pyramid_tokens(
        scores=second_scores,
        active_positions=first_positions,
        original_seq_len=10,
        target_ratio=0.3,
    )

    assert first_positions.tolist() == [1, 2, 3, 4, 5]
    assert second_local_indices.tolist() == [2, 3, 4]
    assert second_positions.tolist() == [3, 4, 5]


def test_pyramid_config_defaults_to_disabled() -> None:
    """Existing configurations should keep the legacy blending behavior."""
    config = LMCacheEngineConfig.from_defaults()

    assert config.blend_pyramid_enabled is False
    assert config.blend_pyramid_beta == 20.0


def test_pyramid_config_accepts_single_start_layer_and_average() -> None:
    """A complete pyramid configuration should pass config validation."""
    config = LMCacheEngineConfig.from_defaults(
        enable_blending=True,
        blend_pyramid_enabled=True,
        blend_pyramid_beta=20,
        blend_check_layers=[1],
        blend_recompute_ratios=[0.16],
    )
    config.validate()

    assert config.blend_check_layers == [1]
    assert config.blend_recompute_ratios == [0.16]


def test_pyramid_plan_caps_custom_selector_budget() -> None:
    """Custom variant selectors should receive a nested per-layer budget."""
    config = LMCacheEngineConfig.from_defaults(
        enable_blending=True,
        blend_pyramid_enabled=True,
        blend_pyramid_beta=3,
        blend_check_layers=[1],
        blend_recompute_ratios=[0.25],
    )
    plan = PyramidRecomputePlan.from_config(config, num_layers=5)

    assert plan.includes(0) is False
    assert plan.includes(1) is True
    assert plan.target_count(100, 100, 1) == 38
    assert plan.target_count(100, 20, 4) == 12


@pytest.mark.parametrize(
    ("check_layers", "ratios"),
    [([], [0.16]), ([1, 15], [0.16]), ([1], []), ([1], [0.16, 0.08])],
)
def test_pyramid_config_rejects_ambiguous_lists(
    check_layers: list[int], ratios: list[float]
) -> None:
    """Pyramid mode should have one start layer and one average ratio."""
    config = LMCacheEngineConfig.from_defaults(
        enable_blending=True,
        blend_pyramid_enabled=True,
        blend_check_layers=check_layers,
        blend_recompute_ratios=ratios,
    )
    with pytest.raises(ValueError):
        config.validate()
