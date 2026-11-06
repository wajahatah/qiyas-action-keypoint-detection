"""Looking around: keypoint-geometry rules with depth-dependent thresholds and a persistence filter.

No classifier is involved — see ``geometry`` (angles, visual-field triangle, field of view),
``thresholds`` (constant / linear / polynomial / field-of-view thresholds) and ``detector``
(per-frame decision, persistence filter, event-level evaluation).
"""

from .detector import (
    METHODS,
    LookingAroundConfig,
    decide,
    detect_sequence,
    detect_table,
    evaluate_table,
    persistence_filter,
    synthetic_layout,
)
from .geometry import (
    centre_angle,
    ear_line_angle,
    field_of_view,
    head_in_visual_field,
    perpendicular_foot,
    point_in_triangle,
)
from .thresholds import (
    DEFAULT_POLY_COEFFS,
    ConstantThreshold,
    FieldOfViewThreshold,
    LinearThreshold,
    PolynomialThreshold,
    depth_proxy,
    make_threshold,
)

__all__ = [
    "DEFAULT_POLY_COEFFS",
    "METHODS",
    "ConstantThreshold",
    "FieldOfViewThreshold",
    "LinearThreshold",
    "LookingAroundConfig",
    "PolynomialThreshold",
    "centre_angle",
    "decide",
    "depth_proxy",
    "detect_sequence",
    "detect_table",
    "ear_line_angle",
    "evaluate_table",
    "field_of_view",
    "head_in_visual_field",
    "make_threshold",
    "perpendicular_foot",
    "persistence_filter",
    "point_in_triangle",
    "synthetic_layout",
]
