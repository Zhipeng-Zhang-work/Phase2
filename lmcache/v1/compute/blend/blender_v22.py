# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache.v1.compute.blend.blender_v21 import (
    _score_documents_with_cross_head_product,
    _SemanticKVShareBlender,
)


class LMCBlender(_SemanticKVShareBlender):
    """V22: V11 scoring with Instruction+Question semantic observers."""

    variant_name = "V22-v11-semantic-instruction-question-observer"

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
        """Apply V11's max-GQA cross-head-product score to I+Q observers."""
        # V22 replaces the old last-N reusable-prefix observation window with
        # the semantic Instruction+Question observer.
        #
        # The old last-N window did not observe the current user query because
        # the legacy CacheBlend path only received the reusable cached prefix.
        return _score_documents_with_cross_head_product(
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
            "max",
        )
