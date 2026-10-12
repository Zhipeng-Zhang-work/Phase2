# SPDX-License-Identifier: Apache-2.0
"""Normalize model output before answer-quality evaluation."""

# Standard
from dataclasses import dataclass
import re

_THINK_END_PATTERN = re.compile(r"</think\s*>", flags=re.IGNORECASE)
_THINK_START_PATTERN = re.compile(r"^\s*<think\s*>", flags=re.IGNORECASE)


@dataclass(frozen=True)
class ProcessedModelOutput:
    """Raw model output and the final answer used for quality evaluation."""

    raw_text: str
    final_answer: str
    reasoning_detected: bool
    incomplete_reasoning: bool


def process_model_output(
    raw_text: str | None,
    reasoning_content: str | None = None,
) -> ProcessedModelOutput:
    """Extract the user-facing answer from one model response.

    Qwen3 normally receives the opening ``<think>`` token from its chat
    template, so leaked reasoning may contain only a generated ``</think>``
    marker. The text after the last closing marker is therefore treated as the
    final answer. A response that starts with ``<think>`` but never closes it
    has no completed final answer.

    Args:
        raw_text: Text returned in the response content field.
        reasoning_content: Optional reasoning returned in a separate API field.
            It is never included in the final answer.

    Returns:
        The preserved raw response, extracted final answer, and reasoning
        detection flags.
    """
    text = "" if raw_text is None else str(raw_text)
    explicit_reasoning = reasoning_content is not None and bool(
        str(reasoning_content).strip()
    )
    closing_markers = list(_THINK_END_PATTERN.finditer(text))
    if closing_markers:
        final_answer = text[closing_markers[-1].end() :].strip()
        return ProcessedModelOutput(
            raw_text=text,
            final_answer=final_answer,
            reasoning_detected=True,
            incomplete_reasoning=False,
        )

    if _THINK_START_PATTERN.match(text):
        return ProcessedModelOutput(
            raw_text=text,
            final_answer="",
            reasoning_detected=True,
            incomplete_reasoning=True,
        )

    return ProcessedModelOutput(
        raw_text=text,
        final_answer=text.strip(),
        reasoning_detected=explicit_reasoning,
        incomplete_reasoning=False,
    )
