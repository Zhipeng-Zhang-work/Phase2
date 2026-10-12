# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for TurboRAG cache position relocation."""

# Third Party
import torch

# First Party
from Baseline.TurboRAG.lmcache.engine import rerotate_key


def test_rerotation_round_trip() -> None:
    """Moving Qwen keys to new positions and back preserves values."""
    torch.manual_seed(0)
    keys = torch.randn(1, 2, 7, 16)
    old_positions = torch.arange(7, dtype=torch.float32)
    new_positions = old_positions + 23
    moved = rerotate_key(keys, old_positions, new_positions, 10000.0)
    restored = rerotate_key(moved, new_positions, old_positions, 10000.0)
    torch.testing.assert_close(restored, keys, atol=1e-5, rtol=1e-5)
