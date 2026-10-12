# SPDX-License-Identifier: Apache-2.0

"""Offline collection helpers for token-level recompute teacher datasets."""

# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import json

# Third Party
import torch

# First Party
from benchmarks.rag.utils import (
    PromptBuildMethodType,
    build_rag_prompt_token_ids,
    query_prompt_set,
    system_prompt_set,
)
from kvrecompute.data import TeacherDatasetTensors


def _normalize_answers(answers: list[Any]) -> list[str]:
    """Flatten nested answer lists into a list of answer strings."""

    normalized_answers: list[str] = []
    for answer in answers:
        if isinstance(answer, list):
            if len(answer) == 1 and isinstance(answer[0], str):
                normalized_answers.append(answer[0])
            else:
                normalized_answers.extend(str(item) for item in answer)
        else:
            normalized_answers.append(str(answer))
    return normalized_answers


@dataclass(frozen=True)
class RecomputeExample:
    """A normalized QA example used by the offline recompute collector.

    Attributes
    ----------
    sample_id:
        Integer identifier inside the dataset split.
    prompt_token_ids:
        Full prompt token ids matching the benchmark formatting.
    doc_prompts:
        Per-document text segments used during standalone document precompute.
    question:
        Raw question string.
    answers:
        Flattened list of answer strings.
    """

    sample_id: int
    prompt_token_ids: list[int]
    doc_prompts: list[str]
    question: str
    answers: list[str]


@dataclass(frozen=True)
class TeacherLabelResult:
    """Teacher outputs for one example.

    Attributes
    ----------
    sample_id:
        Example identifier.
    features:
        Token-level feature matrix of shape ``[num_tokens, num_features]``.
    labels:
        Token-level teacher scores of shape ``[num_tokens]``.
    metadata:
        Optional auxiliary metadata for later inspection.
    """

    sample_id: int
    features: torch.Tensor
    labels: torch.Tensor
    metadata: Optional[dict[str, Any]] = None


class BaseTeacherLabeler:
    """Abstract interface for token-level recompute teacher generation.

    Subclasses are expected to compute, for one normalized example:

    - token-level features available to the online blender
    - token-level scalar labels such as cross-entropy improvement

    The heavy Hugging Face forward logic belongs in subclasses so this module can
    remain lightweight and reusable.
    """

    def collect_example(self, example: RecomputeExample) -> TeacherLabelResult:
        """Collect token-level features and labels for one example.

        Parameters
        ----------
        example:
            Normalized QA example.

        Returns
        -------
        TeacherLabelResult
            Feature and label tensors for the example.
        """

        raise NotImplementedError


class WikiMQACollector:
    """Collector utilities for WikiMQA-style QA datasets."""

    def __init__(
        self,
        tokenizer,
        separator: str = " # # ",
        prompt_build_method: PromptBuildMethodType = PromptBuildMethodType.QA,
    ) -> None:
        self._tokenizer = tokenizer
        self._separator = separator
        self._prompt_build_method = prompt_build_method
        self._system_prompt = system_prompt_set[prompt_build_method]
        self._query_prompt = query_prompt_set[prompt_build_method]

    def load_examples(self, dataset_path: str | Path) -> list[RecomputeExample]:
        """Load and normalize WikiMQA-style examples from JSON.

        Parameters
        ----------
        dataset_path:
            Path to a JSON file containing QA examples with ``ctxs``,
            ``question`` and ``answers`` fields.

        Returns
        -------
        list[RecomputeExample]
            Normalized examples matching benchmark prompt construction.
        """

        dataset_file = Path(dataset_path)
        with dataset_file.open("r", encoding="utf-8") as file:
            raw_examples = json.load(file)

        normalized_examples: list[RecomputeExample] = []
        for sample_id, raw_example in enumerate(raw_examples):
            prompt_token_ids, doc_prompts = build_rag_prompt_token_ids(
                tokenizer=self._tokenizer,
                system_prompt=self._system_prompt,
                example=raw_example,
                query_prompt=self._query_prompt,
                separator=self._separator,
                prompt_build_method=self._prompt_build_method,
            )
            normalized_examples.append(
                RecomputeExample(
                    sample_id=sample_id,
                    prompt_token_ids=prompt_token_ids,
                    doc_prompts=doc_prompts,
                    question=str(raw_example["question"]),
                    answers=_normalize_answers(raw_example["answers"]),
                )
            )

        return normalized_examples

    def build_answer_ce_targets(
        self,
        example: RecomputeExample,
    ) -> list[str]:
        """Return normalized answer strings for CE-based teacher construction.

        Parameters
        ----------
        example:
            Normalized QA example.

        Returns
        -------
        list[str]
            Answer strings that can be used to build teacher-forcing targets.
        """

        return example.answers


def collate_teacher_results(
    results: list[TeacherLabelResult],
) -> TeacherDatasetTensors:
    """Merge per-example teacher results into one token-level dataset.

    Parameters
    ----------
    results:
        Per-example collection outputs.

    Returns
    -------
    TeacherDatasetTensors
        Merged token-level training bundle.
    """

    if not results:
        raise ValueError("No teacher results were provided.")

    features = torch.cat([result.features for result in results], dim=0)
    labels = torch.cat([result.labels for result in results], dim=0)
    metadata = {
        "sample_id": torch.cat(
            [
                torch.full(
                    (result.features.shape[0],),
                    result.sample_id,
                    dtype=torch.int64,
                )
                for result in results
            ]
        )
    }
    return TeacherDatasetTensors(features=features, labels=labels, metadata=metadata)


def save_teacher_dataset(
    teacher_dataset: TeacherDatasetTensors,
    output_path: str | Path,
) -> None:
    """Save a merged teacher dataset to disk.

    Parameters
    ----------
    teacher_dataset:
        Token-level feature/label bundle.
    output_path:
        Destination path for ``torch.save``.
    """

    payload = {
        "features": teacher_dataset.features,
        "labels": teacher_dataset.labels,
        "metadata": teacher_dataset.metadata,
    }
    torch.save(payload, Path(output_path))
