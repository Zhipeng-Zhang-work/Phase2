# SPDX-License-Identifier: Apache-2.0
"""Dataset and prompt handling for standalone TurboRAG evaluation."""

# Standard
from pathlib import Path
from typing import Any
import json

QA_SYSTEM_PROMPT = (
    "You will be asked a question after reading several passages. "
    "Answer directly from the passages in at most five words.\nPassages:\n"
)
SUMMARY_SYSTEM_PROMPT = (
    "Summarize the dialogue into a few short sentences. "
    "The following are examples.\n\n"
)


def load_dataset(path: str) -> list[dict[str, Any]]:
    """Load an evaluation dataset from a JSON array."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("evaluation dataset must be a JSON list")
    return data


def build_example(
    example: dict[str, Any],
    prompt_build_method: str,
) -> tuple[str, list[str], str, list[str]]:
    """Build system, document, query, and reference strings."""
    method = prompt_build_method.upper()
    contexts = example.get("ctxs", [])
    if method == "QA":
        documents = [
            f"{context.get('title', '')}\n\n{context.get('text', '')}\n\n"
            for context in contexts
        ]
        question = str(example["question"])
        if not question.endswith("?"):
            question += "?"
        query = (
            "\nAnswer the question directly based on the passages. "
            "Do not repeat the question. Answer in at most five words."
            f"\nQuestion: {question}\nAnswer:"
        )
        system = QA_SYSTEM_PROMPT
    elif method == "FEW_SHOT":
        documents = [str(context.get("text", "")) for context in contexts]
        query = "\n\n" + str(example["question"])
        system = SUMMARY_SYSTEM_PROMPT
    else:
        raise ValueError(f"unsupported prompt build method: {prompt_build_method}")
    answers = example.get("answers", [])
    references = [answers] if isinstance(answers, str) else list(answers)
    return system, documents, query, [str(answer) for answer in references]
