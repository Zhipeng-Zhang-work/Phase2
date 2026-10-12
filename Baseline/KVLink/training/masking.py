# SPDX-License-Identifier: Apache-2.0
"""KVLink token layout and independent-document attention masks."""

# Standard
from dataclasses import dataclass
from typing import Any

# Third Party
import torch


@dataclass(frozen=True)
class KVLinkTokens:
    """Token IDs required by KVLink training and inference."""

    memory_start: int
    memory_end: int
    link_tokens: tuple[tuple[int, ...], ...]


def register_kvlink_tokens(
    tokenizer: Any,
    max_documents: int,
    link_tokens_per_document: int,
) -> KVLinkTokens:
    """Register KVLink control and link tokens on a tokenizer.

    Args:
        tokenizer: Hugging Face tokenizer to extend.
        max_documents: Maximum number of documents in one sample.
        link_tokens_per_document: Number of link tokens assigned to each
            document.

    Returns:
        IDs of the registered control and link tokens.

    Raises:
        ValueError: If either size argument is not positive.
    """
    if max_documents <= 0:
        raise ValueError("max_documents must be positive")
    if link_tokens_per_document <= 0:
        raise ValueError("link_tokens_per_document must be positive")

    memory_start = "<|kv_start|>"
    memory_end = "<|kv_end|>"
    link_names = [
        f"<|link_{doc_idx}_{token_idx}|>"
        for doc_idx in range(max_documents)
        for token_idx in range(link_tokens_per_document)
    ]
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [memory_start, memory_end, *link_names]}
    )

    link_ids = tuple(
        tuple(
            tokenizer.convert_tokens_to_ids(f"<|link_{doc_idx}_{token_idx}|>")
            for token_idx in range(link_tokens_per_document)
        )
        for doc_idx in range(max_documents)
    )
    return KVLinkTokens(
        memory_start=tokenizer.convert_tokens_to_ids(memory_start),
        memory_end=tokenizer.convert_tokens_to_ids(memory_end),
        link_tokens=link_ids,
    )


def build_kvlink_attention_mask(
    segment_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build KVLink's causal independent-document additive attention mask.

    Positive segment IDs identify document tokens. Document tokens attend only
    to earlier tokens in the same document. Segment zero identifies system,
    link, query, and answer tokens; these tokens attend to every earlier
    non-padding token. Segment ``-1`` is padding.

    Args:
        segment_ids: Integer tensor with shape ``[batch, sequence]``.
        dtype: Floating point dtype of the returned additive mask.

    Returns:
        Additive mask with shape ``[batch, 1, sequence, sequence]``.

    Raises:
        ValueError: If `segment_ids` is not rank two.
    """
    if segment_ids.ndim != 2:
        raise ValueError("segment_ids must have shape [batch, sequence]")

    batch_size, sequence_length = segment_ids.shape
    target = segment_ids.unsqueeze(-1)
    source = segment_ids.unsqueeze(-2)
    same_document = target.eq(source)
    target_is_global = target.eq(0)
    source_is_valid = source.ne(-1)
    target_is_valid = target.ne(-1)
    causal = torch.ones(
        (sequence_length, sequence_length),
        dtype=torch.bool,
        device=segment_ids.device,
    ).tril()
    allowed = (
        (same_document | target_is_global) & source_is_valid & target_is_valid & causal
    )
    mask = torch.full(
        (batch_size, sequence_length, sequence_length),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=segment_ids.device,
    )
    mask.masked_fill_(allowed, 0)
    return mask.unsqueeze(1)
