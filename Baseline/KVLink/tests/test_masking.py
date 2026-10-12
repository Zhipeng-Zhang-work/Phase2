# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for KVLink attention masks."""

# Third Party
import torch

# First Party
from Baseline.KVLink.training.masking import build_kvlink_attention_mask


def test_independent_document_mask() -> None:
    """Document tokens stay isolated while global tokens see prior tokens."""
    segments = torch.tensor([[1, 1, 0, 2, 2, 0, -1]])
    mask = build_kvlink_attention_mask(segments, torch.float32)[0, 0]
    assert mask[1, 0] == 0
    assert mask[1, 2] < -1e20
    assert mask[2, :3].eq(0).all()
    assert mask[4, 0] < -1e20
    assert mask[4, 2] < -1e20
    assert mask[4, 3] == 0
    assert mask[4, 4] == 0
    assert mask[5, :6].eq(0).all()
    assert mask[6].lt(-1e20).all()
