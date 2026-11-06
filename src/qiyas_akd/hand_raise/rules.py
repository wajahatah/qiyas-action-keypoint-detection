"""Geometric hand-raise rule — the first baseline, before any learned model.

An arm counts as raised when the elbow is nearly straight (**elbow angle > 130°**) **and** the wrist
is **above** the same-side shoulder (smaller image y). A person raises a hand when either arm does.

The original script only drew its decisions on test videos — no quantitative evaluation was
recorded — and the rule was superseded by learned keypoint classifiers. It is kept as the reference
baseline the learned models must beat, and its two cues are offered to them as engineered features.

Preserved quirk: the original only evaluated the rule when **all six arm keypoints of both arms**
were detected (``require_both_arms=True``); one occluded arm disabled the rule for the person
entirely. ``require_both_arms=False`` evaluates each arm on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..keypoints import KP
from .features import elbow_angle, missing_mask, table_to_xy

MIN_ELBOW_ANGLE = 130.0
_ARM = ("shoulder", "elbow", "hand")


def arm_raised(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    min_elbow_angle: float = MIN_ELBOW_ANGLE,
) -> bool | None:
    """Rule for one arm; ``None`` when a keypoint is missing."""
    ang = elbow_angle(shoulder, elbow, wrist)
    if ang is None:
        return None
    return ang > min_elbow_angle and float(wrist[1]) < float(shoulder[1])


def hand_raised(
    xy: np.ndarray,
    min_elbow_angle: float = MIN_ELBOW_ANGLE,
    require_both_arms: bool = True,
) -> bool:
    """Rule for one person's keypoints ``(K, 2)`` in pixels."""
    xy = np.asarray(xy, dtype=float)
    arm_idx = [KP[f"{s}_{p}"] for s in ("left", "right") for p in _ARM]
    if require_both_arms and missing_mask(xy[arm_idx]).any():
        return False
    for side in ("left", "right"):
        sh, el, wr = (xy[KP[f"{side}_{p}"]] for p in _ARM)
        if arm_raised(sh, el, wr, min_elbow_angle):
            return True
    return False


def predict_table(
    df: pd.DataFrame,
    min_elbow_angle: float = MIN_ELBOW_ANGLE,
    require_both_arms: bool = True,
) -> np.ndarray:
    """Per-row 0/1 rule predictions for a pixel frame table."""
    return np.array(
        [hand_raised(p, min_elbow_angle, require_both_arms) for p in table_to_xy(df)], dtype=int
    )
