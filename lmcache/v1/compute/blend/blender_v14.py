# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.compute.blend.blender_cacheslide import CacheSlideWeightedBlender


class LMCBlender(CacheSlideWeightedBlender):
    """CacheSlide-WCA-KV: deviation-weighted fusion of keys and values."""

    fuse_values = True
    variant_name = "CacheSlide-WCA-KV"
