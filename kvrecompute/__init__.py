# SPDX-License-Identifier: Apache-2.0

"""Utilities for token-level recompute score training."""

# First Party
from kvrecompute.collector import (
    BaseTeacherLabeler,
    RecomputeExample,
    TeacherLabelResult,
    WikiMQACollector,
    collate_teacher_results,
    save_teacher_dataset,
)
from kvrecompute.data import (
    FEATURE_NAMES,
    TeacherDatasetTensors,
    TokenScoreDataset,
    build_blender_features,
    load_teacher_dataset,
    load_wikimqa_json,
)
from kvrecompute.hf_teacher import (
    PerExampleTeacherTrace,
    compute_gain_labels,
    load_teacher_trace,
    trace_to_teacher_result,
)
from kvrecompute.model import TokenMLPRegressor

__all__ = [
    "BaseTeacherLabeler",
    "FEATURE_NAMES",
    "PerExampleTeacherTrace",
    "RecomputeExample",
    "TeacherDatasetTensors",
    "TeacherLabelResult",
    "TokenMLPRegressor",
    "TokenScoreDataset",
    "WikiMQACollector",
    "build_blender_features",
    "collate_teacher_results",
    "compute_gain_labels",
    "load_teacher_dataset",
    "load_teacher_trace",
    "load_wikimqa_json",
    "save_teacher_dataset",
    "trace_to_teacher_result",
]
