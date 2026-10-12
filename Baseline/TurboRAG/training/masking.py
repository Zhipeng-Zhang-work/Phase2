# SPDX-License-Identifier: Apache-2.0
"""TurboRAG control tokens and independent-document attention masks."""

# Standard
from dataclasses import dataclass
from typing import Any

# Third Party
import torch


@dataclass(frozen=True)
class TurboRAGTokens:
    """Token IDs used to delimit independently cached documents."""

    document_start: int
    document_end: int


def register_turborag_tokens(tokenizer: Any) -> TurboRAGTokens:
    """Register TurboRAG document boundary tokens."""
    start = "<|doc_start|>"
    end = "<|doc_end|>"
    tokenizer.add_special_tokens({"additional_special_tokens": [start, end]})
    return TurboRAGTokens(
        document_start=tokenizer.convert_tokens_to_ids(start),
        document_end=tokenizer.convert_tokens_to_ids(end),
    )


def build_turborag_attention_mask(
    segment_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build TurboRAG's independent-document causal additive mask.

    Document tokens, including boundary tokens, use positive segment IDs and
    can only attend earlier positions in the same document. System, query, and
    answer tokens use segment zero and can attend all earlier valid positions.

    Args:
        segment_ids: Tensor with shape ``[batch, sequence]``.
        dtype: Floating point dtype for the additive mask.

    Returns:
        Mask with shape ``[batch, 1, sequence, sequence]``.
    """
    if segment_ids.ndim != 2:
        raise ValueError("segment_ids must have shape [batch, sequence]")
    batch_size, sequence_length = segment_ids.shape
    target = segment_ids.unsqueeze(-1)
    source = segment_ids.unsqueeze(-2)
    causal = torch.ones(
        (sequence_length, sequence_length),
        dtype=torch.bool,
        device=segment_ids.device,
    ).tril()
    allowed = (
        (target.eq(source) | target.eq(0)) & source.ne(-1) & target.ne(-1) & causal
    )
    mask = torch.full(
        (batch_size, sequence_length, sequence_length),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=segment_ids.device,
    )
    mask.masked_fill_(allowed, 0)
    return mask.unsqueeze(1)
