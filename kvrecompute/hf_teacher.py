# SPDX-License-Identifier: Apache-2.0

"""Teacher-label utilities for token-level recompute gain training."""

# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Third Party
import torch

# First Party
from kvrecompute.collector import TeacherLabelResult


@dataclass(frozen=True)
class PerExampleTeacherTrace:
    """Per-example trace bundle used to derive token-level teacher labels.

    Attributes
    ----------
    sample_id:
        Integer identifier of the source example.
    features:
        Token-level feature tensor of shape ``[num_tokens, num_features]``.
    ce_base:
        Baseline answer cross-entropy under direct cache reuse.
    ce_per_token:
        Cross-entropy after recomputing exactly one token at a time. Must have
        shape ``[num_tokens]`` and align with ``features``.
    metadata:
        Optional free-form metadata for debugging or later analysis.
    """

    sample_id: int
    features: torch.Tensor
    ce_base: torch.Tensor
    ce_per_token: torch.Tensor
    metadata: Optional[dict[str, Any]] = None


def compute_gain_labels(
    ce_base: torch.Tensor,
    ce_per_token: torch.Tensor,
) -> torch.Tensor:
    """Compute token-level recompute gain labels from CE traces.

    Parameters
    ----------
    ce_base:
        Scalar tensor containing the baseline answer CE under direct reuse.
    ce_per_token:
        Tensor of shape ``[num_tokens]`` containing answer CE after
        recomputing each token individually.

    Returns
    -------
    torch.Tensor
        Gain label tensor of shape ``[num_tokens]`` where larger values mean
        greater expected quality improvement from recomputing that token.
    """

    if ce_base.ndim != 0:
        raise ValueError(f"ce_base must be scalar, got shape {tuple(ce_base.shape)}")
    if ce_per_token.ndim != 1:
        raise ValueError(
            f"ce_per_token must be 1D, got shape {tuple(ce_per_token.shape)}"
        )
    return ce_base.to(torch.float32) - ce_per_token.to(torch.float32)


def load_teacher_trace(trace_path: str | Path) -> PerExampleTeacherTrace:
    """Load one serialized per-example CE trace bundle.

    The trace file must be saved with ``torch.save`` and contain:

    - ``sample_id``: integer
    - ``features``: ``[num_tokens, num_features]``
    - ``ce_base``: scalar tensor or float
    - ``ce_per_token``: ``[num_tokens]``
    - optional ``metadata``

    Parameters
    ----------
    trace_path:
        Path to a per-example trace file.

    Returns
    -------
    PerExampleTeacherTrace
        Parsed trace bundle.
    """

    payload = torch.load(Path(trace_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Teacher trace must be a dictionary saved with torch.save")

    sample_id = int(payload["sample_id"])
    features = payload["features"]
    ce_base = torch.as_tensor(payload["ce_base"])
    ce_per_token = torch.as_tensor(payload["ce_per_token"])
    metadata = payload.get("metadata")

    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {tuple(features.shape)}")
    if ce_per_token.ndim != 1:
        raise ValueError(
            f"ce_per_token must be 1D, got shape {tuple(ce_per_token.shape)}"
        )
    if features.shape[0] != ce_per_token.shape[0]:
        raise ValueError(
            "features and ce_per_token must align on num_tokens, got "
            f"{features.shape[0]} vs {ce_per_token.shape[0]}"
        )

    return PerExampleTeacherTrace(
        sample_id=sample_id,
        features=features,
        ce_base=ce_base,
        ce_per_token=ce_per_token,
        metadata=metadata,
    )


def trace_to_teacher_result(
    trace: PerExampleTeacherTrace,
) -> TeacherLabelResult:
    """Convert one CE trace bundle into token-level training labels.

    Parameters
    ----------
    trace:
        Per-example CE trace bundle.

    Returns
    -------
    TeacherLabelResult
        Token-level feature tensor and gain labels.
    """

    labels = compute_gain_labels(trace.ce_base, trace.ce_per_token)
    return TeacherLabelResult(
        sample_id=trace.sample_id,
        features=trace.features.to(torch.float32),
        labels=labels.to(torch.float32),
        metadata=trace.metadata,
    )

