"""Pose model tooling: dataset preparation, fine-tuning, export, benchmarking and inference.

Importing this package never imports Ultralytics or OpenCV; those are loaded lazily by the
functions that need them (install with ``pip install 'qiyas-akd[pose]'``).
"""

from .dataset import (
    ANNOTATION_ORDER,
    FLIP_IDX,
    build_data_yaml,
    compute_flip_idx,
    plan_video_split,
    reorder_label_dir,
    reorder_label_line,
    split_by_video,
    video_key,
    write_data_yaml,
)
from .infer import PoseDetections, UltralyticsPoseModel, detections_from_array

__all__ = [
    "ANNOTATION_ORDER",
    "FLIP_IDX",
    "PoseDetections",
    "UltralyticsPoseModel",
    "build_data_yaml",
    "compute_flip_idx",
    "detections_from_array",
    "plan_video_split",
    "reorder_label_dir",
    "reorder_label_line",
    "split_by_video",
    "video_key",
    "write_data_yaml",
]
