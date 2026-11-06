"""Per-person hand-raise features from one frame of 10 keypoints.

Layout: an **x block followed by a y block** — ``[x_0 … x_9, y_0 … y_9]`` in ``KEYPOINT_NAMES``
order. That is the layout the two-branch MLP consumes (x-branch / y-branch) and the layout the
random forest was trained on, so all models share one feature matrix.

Variants:

* ``normalise`` — ``x / 1280``, ``y / 720``, rounded to 3 decimals (``keypoints.normalize_xy``).
  Several original preprocessing and inference scripts divided y by **780** instead of the 720-px
  frame height, while others used 720; one divisor (720) is used everywhere here.
* ``generalise`` — the x coordinate is first shifted by the left edge of the candidate's seat ROI:
  ``(x − roi_left) / 1280``. The aim was to make a candidate at desk 4 look like one at desk 1 so
  the classifier learns the pose, not the seat. Note this is a *shift*, not
  ``keypoints.roi_relative`` (which also rescales by the ROI width).
* ``engineered`` (optional extra columns) — elbow angle per arm (scaled to ``[0, 1]`` by 180°) and a
  wrist-above-shoulder flag per arm: the two cues of the geometric rule (``rules.py``), offered to
  the learned models as explicit features.

Missing keypoints (pose stage ``(0, 0)``) become ``MISSING`` (-1) by default. The original scripts
used ``(0, 0)`` for missing normalised points; pass ``missing_value=0.0`` to reproduce that
encoding.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..desks import CameraLayout
from ..keypoints import FRAME_H, FRAME_W, KP, MISSING, NUM_KEYPOINTS, ROUND_DECIMALS, normalize_xy

VARIANTS = ("normalise", "generalise")
X_COLUMNS = [f"kp_{k}_x" for k in range(NUM_KEYPOINTS)]
Y_COLUMNS = [f"kp_{k}_y" for k in range(NUM_KEYPOINTS)]
FEATURE_NAMES = X_COLUMNS + Y_COLUMNS
ENGINEERED_NAMES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_wrist_above_shoulder",
    "right_wrist_above_shoulder",
]


def missing_mask(xy: np.ndarray) -> np.ndarray:
    """True where a pixel keypoint is missing: ``(0, 0)`` or any NaN coordinate."""
    return np.isnan(xy).any(axis=-1) | (xy == 0).all(axis=-1)


def table_to_xy(df: pd.DataFrame) -> np.ndarray:
    """Pixel keypoints ``(N, K, 2)`` from a frame table with ``kp_{k}_x`` / ``kp_{k}_y`` columns."""
    cols = [f"kp_{k}_{c}" for k in range(NUM_KEYPOINTS) for c in ("x", "y")]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"frame table lacks keypoint columns {missing[:3]}…")
    return df[cols].to_numpy(dtype=float).reshape(len(df), NUM_KEYPOINTS, 2)


def _blocks(norm: np.ndarray) -> np.ndarray:
    """``(N, K, 2)`` → ``(N, 2K)`` as ``[x block, y block]``."""
    return np.concatenate([norm[..., 0], norm[..., 1]], axis=-1)


def normalised_features(xy: np.ndarray, missing_value: float = MISSING) -> np.ndarray:
    """``(N, K, 2)`` (or ``(K, 2)``) pixels → ``(N, 20)`` frame-normalised features."""
    xy = np.asarray(xy, dtype=float)
    single = xy.ndim == 2
    xy = xy[None] if single else xy
    norm = normalize_xy(xy)
    norm[missing_mask(xy)] = missing_value
    out = _blocks(norm)
    return out[0] if single else out


def generalised_features(
    xy: np.ndarray, roi_left: float | Sequence[float] | np.ndarray, missing_value: float = MISSING
) -> np.ndarray:
    """``(N, K, 2)`` pixels → ``(N, 20)`` with x shifted by each row's seat-ROI left edge.

    ``roi_left`` is a scalar or one value per row.
    """
    xy = np.asarray(xy, dtype=float)
    single = xy.ndim == 2
    xy = xy[None] if single else xy
    left = np.broadcast_to(np.asarray(roi_left, dtype=float), (xy.shape[0],))
    out = np.empty_like(xy)
    out[..., 0] = (xy[..., 0] - left[:, None]) / FRAME_W
    out[..., 1] = xy[..., 1] / FRAME_H
    out = np.round(out, ROUND_DECIMALS)
    out[missing_mask(xy)] = missing_value
    res = _blocks(out)
    return res[0] if single else res


def elbow_angle(
    shoulder: Sequence[float], elbow: Sequence[float], wrist: Sequence[float]
) -> float | None:
    """Interior angle at the elbow (degrees, 0–180) between elbow→shoulder and elbow→wrist.

    180° is a fully straight arm. Returns ``None`` if a point is missing or a segment has zero
    length. Only ``(x, y)`` are used — the original passed the pose model's ``(x, y, confidence)``
    rows straight into the dot product, so confidence differences leaked into the "angle". The
    cosine is clipped to ``[-1, 1]`` to avoid ``NaN`` from rounding on straight arms.
    """
    pts = np.array([shoulder, elbow, wrist], dtype=float)[:, :2]
    if missing_mask(pts).any():
        return None
    v1, v2 = pts[0] - pts[1], pts[2] - pts[1]
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return None
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def engineered_features(xy: np.ndarray, missing_value: float = MISSING) -> np.ndarray:
    """``(N, K, 2)`` pixels → ``(N, 4)``: elbow angle / 180 per arm, wrist-above-shoulder per arm.

    A missing arm gives ``missing_value`` for its angle and 0 for its flag. "Above" means a smaller
    image y than the same-side shoulder.
    """
    xy = np.asarray(xy, dtype=float)
    single = xy.ndim == 2
    xy = xy[None] if single else xy
    out = np.zeros((xy.shape[0], 4))
    for i, pose in enumerate(xy):
        for j, side in enumerate(("left", "right")):
            sh, el, wr = (pose[KP[f"{side}_{p}"]] for p in ("shoulder", "elbow", "hand"))
            ang = elbow_angle(sh, el, wr)
            out[i, j] = missing_value if ang is None else ang / 180.0
            out[i, 2 + j] = float(ang is not None and wr[1] < sh[1])
    return out[0] if single else out


def roi_left_from_layout(
    df: pd.DataFrame, layout: CameraLayout, desk_column: str = "desk_no"
) -> np.ndarray:
    """Seat-ROI left edge for every row of a frame table, looked up by desk number."""
    lefts = {d.desk_id: float(d.roi[0]) for d in layout.desks}
    unknown = set(df[desk_column].astype(int)) - set(lefts)
    if unknown:
        raise KeyError(f"desks {sorted(unknown)} not in camera layout {layout.camera_id!r}")
    return df[desk_column].astype(int).map(lefts).to_numpy(dtype=float)


def build_features(
    df: pd.DataFrame,
    variant: str = "normalise",
    engineered: bool = False,
    roi_left: np.ndarray | None = None,
    missing_value: float = MISSING,
) -> np.ndarray:
    """Feature matrix ``(N, 20)`` (or ``(N, 24)`` with ``engineered``) from a pixel frame table."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    xy = table_to_xy(df)
    if variant == "normalise":
        feats = normalised_features(xy, missing_value)
    else:
        if roi_left is None:
            raise ValueError("the 'generalise' variant needs roi_left (see roi_left_from_layout)")
        feats = generalised_features(xy, roi_left, missing_value)
    if engineered:
        feats = np.concatenate([feats, engineered_features(xy, missing_value)], axis=1)
    return feats.astype(np.float32)


def feature_names(engineered: bool = False) -> list[str]:
    """Column names matching ``build_features``."""
    return FEATURE_NAMES + (ENGINEERED_NAMES if engineered else [])
