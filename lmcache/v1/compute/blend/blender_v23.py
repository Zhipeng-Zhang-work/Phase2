# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache.v1.compute.blend.blender_v21 import (
    _grouped_semantic_attention_and_value_deviation,
    _SemanticKVShareBlender,
)


class LMCBlender(_SemanticKVShareBlender):
    """V23: V12 head-aligned scoring with semantic I+Q observers."""

    variant_name = "V23-v12-semantic-instruction-question-observer"

    def _score_documents(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        cached_values: torch.Tensor,
        observer_indices: torch.Tensor,
        document_indices: torch.Tensor,
        positions: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Apply V12's mean-GQA head-aligned score to I+Q observers."""
        # V23 replaces the old last-N reusable-prefix observation window with
        # the semantic Instruction+Question observer.
        #
        # The old last-N window did not observe the current user query because
        # the legacy CacheBlend path only received the reusable cached prefix.
        document_attention, value_deviation = (
            _grouped_semantic_attention_and_value_deviation(
                queries,
                keys,
                values,
                cached_values,
                observer_indices,
                document_indices,
                positions,
                num_heads,
                num_kv_heads,
                head_dim,
                "mean",
            )
        )
        attention_per_kv_head = document_attention.sum(dim=1)
        document_scores = (attention_per_kv_head * value_deviation).sum(dim=0)
        if document_scores.numel() != document_indices.numel():
            raise AssertionError("Document score count does not match candidates")
        return document_scores
