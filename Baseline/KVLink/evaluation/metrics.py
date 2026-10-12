# SPDX-License-Identifier: Apache-2.0
"""Quality metrics for standalone KVLink evaluation."""

# Standard
from typing import Any
import collections
import re
import string

# Third Party
from rouge_score import rouge_scorer


def strip_reasoning(text: str) -> str:
    """Remove a DeepSeek reasoning trace from generated text."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def compute_quality(
    prediction: str,
    references: list[str],
    method: str,
    tokenizer: Any,
) -> float:
    """Return the best quality score across valid references."""
    if not references:
        return 0.0
    cleaned = strip_reasoning(prediction)
    if method.upper() == "QA":
        return max(_token_f1(cleaned, answer, tokenizer) for answer in references)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return max(
        scorer.score(answer, cleaned)["rougeL"].fmeasure for answer in references
    )


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(
        character for character in text if character not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _token_f1(prediction: str, reference: str, tokenizer: Any) -> float:
    prediction = prediction.lstrip("\n").split("\n")[0]
    prediction_tokens = tokenizer.encode(
        _normalize(prediction),
        add_special_tokens=False,
    )
    reference_tokens = tokenizer.encode(
        _normalize(reference),
        add_special_tokens=False,
    )
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    common = collections.Counter(prediction_tokens) & collections.Counter(
        reference_tokens
    )
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)
