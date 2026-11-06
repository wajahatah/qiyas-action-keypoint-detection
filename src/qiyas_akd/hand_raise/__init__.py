"""Hand raise: geometric rule baseline, keypoint MLPs / random forest, hybrid image+keypoint model.

``features`` builds the shared per-person feature matrix, ``rules`` the elbow-angle rule,
``models`` the model definitions (Keras MLPs ported to PyTorch), ``data`` safe CSV loading and label
validation, ``train`` the config/CLI training with video-grouped evaluation (not imported here so
``python -m qiyas_akd.hand_raise.train`` runs cleanly).
"""

from .data import clean_columns, load_tables, parse_point_columns, validate_binary_labels
from .features import (
    build_features,
    elbow_angle,
    engineered_features,
    feature_names,
    generalised_features,
    normalised_features,
    roi_left_from_layout,
)
from .models import (
    RF_PARAM_GRID,
    HybridModel,
    KeypointMLP,
    TwoBranchMLP,
    grid_search_random_forest,
    make_random_forest,
    multilabel_accuracy,
)
from .rules import arm_raised, hand_raised, predict_table

__all__ = [
    "RF_PARAM_GRID",
    "HybridModel",
    "KeypointMLP",
    "TwoBranchMLP",
    "arm_raised",
    "build_features",
    "clean_columns",
    "elbow_angle",
    "engineered_features",
    "feature_names",
    "generalised_features",
    "grid_search_random_forest",
    "hand_raised",
    "load_tables",
    "make_random_forest",
    "multilabel_accuracy",
    "normalised_features",
    "parse_point_columns",
    "predict_table",
    "roi_left_from_layout",
    "validate_binary_labels",
]
