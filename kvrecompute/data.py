# SPDX-License-Identifier: Apache-2.0

"""Data utilities for token-level recompute score training."""

# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import json

# Third Party
import torch
from torch.utils.data import Dataset


FEATURE_NAMES = (
    "k_l2_diff",
    "v_l2_diff",
    "residual_l2",
    "delta_pos",
)


@dataclass(frozen=True)
class TeacherDatasetTensors:
    """Container for token-level recompute teacher tensors.

    Attributes
    ----------
    features:
        Feature tensor with shape ``[num_tokens, num_features]``.
    labels:
        Scalar target tensor with shape ``[num_tokens]``.
    metadata:
        Optional per-token metadata dictionary. Typical keys include
        ``sample_id``, ``token_index``, ``layer_id`` and ``answer_text``.
    """

    features: torch.Tensor
    labels: torch.Tensor
    metadata: Optional[dict[str, Any]] = None


def load_wikimqa_json(dataset_path: str | Path) -> list[dict[str, Any]]:
    """Load a locally prepared WikiMQA-style dataset.

    Parameters
    ----------
    dataset_path:
        Path to a JSON file containing a list of examples with keys such as
        ``ctxs``, ``question`` and ``answers``.

    Returns
    -------
    list[dict[str, Any]]
        Loaded dataset records.
    """

    path = Path(dataset_path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset at {path}, got {type(data).__name__}")
    return data


def build_blender_features(
    k: torch.Tensor,
    old_k: torch.Tensor,
    v: torch.Tensor,
    old_v: torch.Tensor,
    residual: torch.Tensor,
    delta_pos: int | float | torch.Tensor,
) -> torch.Tensor:
    """Build token-level features available inside ``blender.process_qkv``.

    Parameters
    ----------
    k:
        Current K tensor with shape ``[num_tokens, kv_hidden_dim]``.
    old_k:
        Reused K tensor with shape ``[num_tokens, kv_hidden_dim]``.
    v:
        Current V tensor with shape ``[num_tokens, kv_hidden_dim]``.
    old_v:
        Reused V tensor with shape ``[num_tokens, kv_hidden_dim]``.
    residual:
        Residual tensor with shape ``[num_tokens, hidden_dim]``.
    delta_pos:
        Position shift between the cached chunk and the online prompt. Can be a
        scalar or a tensor broadcastable to ``[num_tokens]``.

    Returns
    -------
    torch.Tensor
        Feature tensor with shape ``[num_tokens, 4]`` ordered according to
        :data:`FEATURE_NAMES`.
    """

    k_l2_diff = torch.linalg.vector_norm(k - old_k, dim=-1)
    v_l2_diff = torch.linalg.vector_norm(v - old_v, dim=-1)
    residual_l2 = torch.linalg.vector_norm(residual, dim=-1)

    if not isinstance(delta_pos, torch.Tensor):
        delta_pos = torch.full_like(k_l2_diff, float(delta_pos))
    else:
        delta_pos = delta_pos.to(device=k.device, dtype=k.dtype).reshape(-1)
        if delta_pos.numel() == 1:
            delta_pos = delta_pos.expand_as(k_l2_diff)

    return torch.stack(
        [
            k_l2_diff.to(torch.float32),
            v_l2_diff.to(torch.float32),
            residual_l2.to(torch.float32),
            delta_pos.to(torch.float32),
        ],
        dim=-1,
    )


def load_teacher_dataset(dataset_path: str | Path) -> TeacherDatasetTensors:
    """Load a precomputed token-level teacher dataset from ``torch.save`` output.

    The saved object must be a dictionary with:

    - ``features``: tensor of shape ``[num_tokens, num_features]``
    - ``labels``: tensor of shape ``[num_tokens]``
    - optional ``metadata``: free-form dictionary

    Parameters
    ----------
    dataset_path:
        Path to the serialized dataset file.

    Returns
    -------
    TeacherDatasetTensors
        Loaded tensor bundle.
    """

    bundle = torch.load(Path(dataset_path), map_location="cpu")
    if not isinstance(bundle, dict):
        raise ValueError("Teacher dataset must be a dictionary saved with torch.save")

    features = bundle["features"]
    labels = bundle["labels"]
    metadata = bundle.get("metadata")

    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {tuple(features.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {tuple(labels.shape)}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "features and labels must agree on num_tokens, got "
            f"{features.shape[0]} vs {labels.shape[0]}"
        )

    return TeacherDatasetTensors(features=features, labels=labels, metadata=metadata)


class TokenScoreDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset of token-level features and scalar recompute labels."""

    def __init__(self, tensors: TeacherDatasetTensors) -> None:
        self._features = tensors.features.to(torch.float32)
        self._labels = tensors.labels.to(torch.float32)

    def __len__(self) -> int:
        """Return the number of token-level training samples."""

        return self._features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one token-level training sample."""

        return self._features[index], self._labels[index]

