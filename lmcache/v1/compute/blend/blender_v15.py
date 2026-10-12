# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.compute.blend.blender_query_aware import QueryAwareDocumentBlender


class LMCBlender(QueryAwareDocumentBlender):
    """V15: instruction-and-question-guided document KV repair. 修正了v4 SnapKV的打分方式"""

    include_instruction_observers = True
    variant_name = "V15-instruction-question-guided-document-repair"
