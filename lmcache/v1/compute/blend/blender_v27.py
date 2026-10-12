# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache.v1.compute.blend.blender_v25 import _KeyAwareSemanticKVShareBlender
from lmcache.v1.compute.blend.blender_v26 import (
    _score_documents_by_dual_attention_shift,
)


class LMCBlender(_KeyAwareSemanticKVShareBlender):
    """V27: token-level Jensen-Shannon contribution times Value mismatch."""

    variant_name = "V27-js-contribution-value-mismatch"

    def _score_documents_key_aware(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        cached_keys: torch.Tensor,
        values: torch.Tensor,
        cached_values: torch.Tensor,
        observer_indices: torch.Tensor,
        document_indices: torch.Tensor,
        positions: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Apply the V27 token-level JS-contribution score."""
        return _score_documents_by_dual_attention_shift(
            queries,
            keys,
            cached_keys,
            values,
            cached_values,
            observer_indices,
            document_indices,
            positions,
            num_heads,
            num_kv_heads,
            head_dim,
            "js_contribution",
        )
