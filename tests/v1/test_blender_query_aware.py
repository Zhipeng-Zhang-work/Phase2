# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1.compute.blend.blender_query_aware import (
    BlendPromptSpans,
    score_documents_by_observer_attention,
    select_document_repair_and_active_positions,
)
from lmcache.v1.compute.blend.blender_v15 import LMCBlender as V15Blender
from lmcache.v1.compute.blend.blender_v16 import LMCBlender as V16Blender


def test_document_scores_preserve_full_softmax_denominator() -> None:
    """Document attention must not be renormalized after column selection."""
    queries = torch.ones((3, 1), dtype=torch.float32)
    keys = torch.zeros((3, 1), dtype=torch.float32)
    scores = score_documents_by_observer_attention(
        queries=queries,
        keys=keys,
        observer_indices=torch.tensor([2]),
        document_indices=torch.tensor([1]),
        positions=torch.arange(3),
        num_heads=1,
        num_kv_heads=1,
        head_dim=1,
    )
    assert scores.item() == pytest.approx(1.0 / 3.0)


def test_document_scores_apply_causal_mask_at_true_positions() -> None:
    """An observer cannot assign attention to a future document position."""
    queries = torch.ones((3, 2), dtype=torch.float32)
    keys = torch.zeros((3, 1), dtype=torch.float32)
    scores = score_documents_by_observer_attention(
        queries=queries,
        keys=keys,
        observer_indices=torch.tensor([1]),
        document_indices=torch.tensor([0, 2]),
        positions=torch.arange(3),
        num_heads=2,
        num_kv_heads=1,
        head_dim=1,
    )
    torch.testing.assert_close(scores, torch.tensor([0.5, 0.0]))


def test_prompt_spans_reject_overlap_and_incomplete_coverage() -> None:
    """Semantic regions must form an exact partition of the prompt."""
    spans = {
        "system_indices": [0],
        "document_indices": [1],
        "separator_indices": [2],
        "instruction_indices": [3],
        "question_indices": [3],
        "answer_prefix_indices": [],
    }
    with pytest.raises(ValueError, match="overlaps"):
        BlendPromptSpans.from_dict(spans, prompt_length=5, device=torch.device("cpu"))


def test_document_budget_excludes_mandatory_online_tokens() -> None:
    """The ratio applies only to reused documents, not the active set."""
    selected, active = select_document_repair_and_active_positions(
        document_scores=torch.arange(10, dtype=torch.float32),
        document_indices=torch.arange(2, 12),
        mandatory_indices=torch.tensor([0, 1, 12, 13, 14]),
        recompute_ratio=0.2,
        prompt_length=15,
    )
    torch.testing.assert_close(selected, torch.tensor([10, 11]))
    assert active.numel() == 7
    assert torch.all(torch.isin(torch.tensor([0, 1, 12, 13, 14]), active))


def test_v15_v16_differ_only_in_instruction_observer_flag() -> None:
    """The two variants are the intended one-variable ablation."""
    assert V15Blender.include_instruction_observers is True
    assert V16Blender.include_instruction_observers is False
    assert V15Blender.__mro__[1:] == V16Blender.__mro__[1:]
