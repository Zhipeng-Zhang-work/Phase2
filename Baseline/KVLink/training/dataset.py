# SPDX-License-Identifier: Apache-2.0
"""JSONL dataset and collator for KVLink fine-tuning."""

# Standard
from pathlib import Path
from typing import Any
import json

# Third Party
from torch.utils.data import Dataset
import torch

# Local
from .masking import KVLinkTokens, build_kvlink_attention_mask

SYSTEM_PROMPT = (
    "Use the supplied documents to answer the question accurately. "
    "Do not use unsupported information."
)


class KVLinkDataset(Dataset[dict[str, Any]]):
    """Materialize KVLink fine-tuning examples from a JSONL file."""

    def __init__(
        self,
        path: str,
        tokenizer: Any,
        tokens: KVLinkTokens,
        max_length: int,
        max_documents: int,
    ) -> None:
        """Load and tokenize training examples.

        Args:
            path: JSONL file following `Baseline/README.md`.
            tokenizer: Hugging Face tokenizer.
            tokens: Registered KVLink token IDs.
            max_length: Maximum sequence length.
            max_documents: Maximum documents retained per example.

        Raises:
            ValueError: If an input row is missing required fields.
        """
        self.examples: list[dict[str, Any]] = []
        input_path = Path(path)
        with input_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                for field in ("documents", "question", "answer"):
                    if field not in row:
                        raise ValueError(
                            f"{input_path}:{line_number} is missing {field}"
                        )
                encoded = self._encode(
                    row,
                    tokenizer,
                    tokens,
                    max_length,
                    max_documents,
                )
                if encoded is not None:
                    self.examples.append(encoded)

    def __len__(self) -> int:
        """Return the number of retained examples."""
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one tokenized example."""
        return self.examples[index]

    @staticmethod
    def _encode(
        row: dict[str, Any],
        tokenizer: Any,
        tokens: KVLinkTokens,
        max_length: int,
        max_documents: int,
    ) -> dict[str, Any] | None:
        system_ids = tokenizer(
            SYSTEM_PROMPT,
            add_special_tokens=True,
        ).input_ids
        question_ids = tokenizer(
            f"\nQuestion: {row['question']}\nAnswer:",
            add_special_tokens=False,
        ).input_ids
        answer_ids = tokenizer(
            f" {row['answer']}",
            add_special_tokens=False,
        ).input_ids
        if tokenizer.eos_token_id is not None:
            answer_ids.append(tokenizer.eos_token_id)

        input_ids = [*system_ids, tokens.memory_start]
        segment_ids = [0] * len(input_ids)
        documents = list(row["documents"])[:max_documents]
        for doc_idx, document in enumerate(documents):
            doc_ids = tokenizer(
                str(document),
                add_special_tokens=False,
            ).input_ids
            input_ids.extend(doc_ids)
            segment_ids.extend([doc_idx + 1] * len(doc_ids))
            link_ids = list(tokens.link_tokens[doc_idx])
            input_ids.extend(link_ids)
            segment_ids.extend([0] * len(link_ids))

        prefix_ids = [tokens.memory_end, *question_ids]
        answer_start = len(input_ids) + len(prefix_ids)
        input_ids.extend(prefix_ids)
        input_ids.extend(answer_ids)
        segment_ids.extend([0] * (len(prefix_ids) + len(answer_ids)))
        if len(input_ids) > max_length:
            return None

        labels = [-100] * answer_start + input_ids[answer_start:]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids,
        }


class KVLinkCollator:
    """Pad KVLink samples and construct their four-dimensional mask."""

    def __init__(self, pad_token_id: int, mask_dtype: torch.dtype) -> None:
        """Configure padding and additive-mask dtype."""
        self.pad_token_id = pad_token_id
        self.mask_dtype = mask_dtype

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate tokenized rows into tensors."""
        max_length = max(len(row["input_ids"]) for row in rows)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        segments: list[list[int]] = []
        for row in rows:
            padding = max_length - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * padding)
            labels.append(row["labels"] + [-100] * padding)
            segments.append(row["segment_ids"] + [-1] * padding)

        segment_tensor = torch.tensor(segments, dtype=torch.long)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "position_ids": torch.arange(max_length, dtype=torch.long)
            .unsqueeze(0)
            .expand(len(rows), -1),
            "attention_mask": build_kvlink_attention_mask(
                segment_tensor,
                self.mask_dtype,
            ),
        }
