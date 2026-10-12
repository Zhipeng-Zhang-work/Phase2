# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.compute.blend.blender_cacheslide import CacheSlideWeightedBlender


class LMCBlender(CacheSlideWeightedBlender):
    """CacheSlide-WCA-K: deviation-weighted fusion of keys only."""

    fuse_values = False
    variant_name = "CacheSlide-WCA-K"
