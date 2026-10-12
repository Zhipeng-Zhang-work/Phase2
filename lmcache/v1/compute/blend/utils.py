# SPDX-License-Identifier: Apache-2.0
# Standard
import os
import re
from importlib import import_module
from typing import TYPE_CHECKING, Dict

# Third Party
from torch import nn

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.blend.blender import LMCBlender
from lmcache.v1.compute.models.utils import VLLMModelTracker

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.gpu_connector import GPUConnectorInterface

logger = init_logger(__name__)


class LMCBlenderBuilder:
    _blenders: Dict[str, LMCBlender] = {}

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        cache_engine: "LMCacheEngine",
        gpu_connector: "GPUConnectorInterface",
        config: "LMCacheEngineConfig",
    ):
        """
        Get or create a blender for the given instance_id.
        """

        if instance_id not in cls._blenders:
            variant = os.environ.get("LMCACHE_BLENDER_VARIANT", "blender")
            if variant != "blender" and re.fullmatch(
                r"blender_v[1-9][0-9]*", variant
            ) is None:
                raise ValueError(f"Invalid LMCACHE_BLENDER_VARIANT: {variant!r}")
            blender_class = import_module(
                f"lmcache.v1.compute.blend.{variant}"
            ).LMCBlender
            logger.info("Creating blender for %s using %s", instance_id, variant)
            vllm_model = VLLMModelTracker.get_model(instance_id)
            blender = blender_class(
                cache_engine=cache_engine,
                gpu_connector=gpu_connector,
                vllm_model=vllm_model,
                config=config,
            )
            cls._blenders[instance_id] = blender
        else:
            logger.info(
                f"Blender for {instance_id} already exists, returning the original one."
            )
        return cls._blenders[instance_id]

    @classmethod
    def get(
        cls,
        instance_id: str,
    ) -> nn.Module:
        """
        Get the blender by instance_id.
        """
        if instance_id not in cls._blenders:
            raise ValueError(f"Blender for {instance_id} not found.")
        return cls._blenders[instance_id]
