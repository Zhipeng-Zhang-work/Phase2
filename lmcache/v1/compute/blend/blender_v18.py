# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache.v1.compute.blend.blender_v17 import LMCBlender as V17LMCBlender


class LMCBlender(V17LMCBlender):
    """V18: V17 scoring with an instruction-and-question observer window."""

    variant_name = "V18-instruction-question-vector-preserving-repair"

    def _observer_indices(self) -> torch.Tensor:
        """Observe first-order output effects at instruction and question rows."""
        assert self.prompt_spans is not None
        return torch.cat(
            (
                self.prompt_spans.instruction_indices,
                self.prompt_spans.question_indices,
            )
        )
