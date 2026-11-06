"""Custom 10-keypoint upper-body schema and the normalisation conventions used by every model.

The pose model was fine-tuned on a custom skeleton (not COCO-17): exam candidates are seen from
an elevated camera, seated behind desks, so legs are never visible and the ears/neck matter more
than hips. Every downstream feature is built from these 10 points.

Conventions (identical across training and inference):

* frames are resized to ``FRAME_W x FRAME_H`` (1280 x 720) before pose inference;
* a keypoint whose confidence is below ``CONF_THRESHOLD`` is treated as *missing*;
* missing keypoints are encoded as ``(0, 0)`` in pixel space and as ``MISSING`` (-1) once
  normalised;
* normalised coordinates are ``x / FRAME_W`` and ``y / FRAME_H``, rounded to 3 decimals.
"""

from __future__ import annotations

import numpy as np

KEYPOINT_NAMES: tuple[str, ...] = (
    "head",
    "neck",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "left_elbow",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_hand",
)
NUM_KEYPOINTS = len(KEYPOINT_NAMES)
KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# Bones used for drawing and for bone-length ("distance") features.
SKELETON: tuple[tuple[int, int], ...] = (
    (KP["head"], KP["neck"]),
    (KP["head"], KP["left_ear"]),
    (KP["head"], KP["right_ear"]),
    (KP["neck"], KP["left_shoulder"]),
    (KP["left_shoulder"], KP["left_elbow"]),
    (KP["left_elbow"], KP["left_hand"]),
    (KP["neck"], KP["right_shoulder"]),
    (KP["right_shoulder"], KP["right_elbow"]),
    (KP["right_elbow"], KP["right_hand"]),
)

FRAME_W = 1280
FRAME_H = 720
CONF_THRESHOLD = 0.5
MISSING = -1.0
ROUND_DECIMALS = 3


def mask_low_confidence(
    xy: np.ndarray, conf: np.ndarray, threshold: float = CONF_THRESHOLD
) -> np.ndarray:
    """Return a copy of ``xy`` (..., K, 2) with low-confidence points set to (0, 0)."""
    xy = np.asarray(xy, dtype=np.float64).copy()
    conf = np.asarray(conf, dtype=np.float64)
    if xy.shape[:-1] != conf.shape:
        raise ValueError(f"xy shape {xy.shape} does not match conf shape {conf.shape}")
    xy[conf < threshold] = 0.0
    return xy


def normalize_xy(
    xy: np.ndarray,
    width: float = FRAME_W,
    height: float = FRAME_H,
    decimals: int | None = ROUND_DECIMALS,
) -> np.ndarray:
    """Normalise pixel keypoints (..., K, 2) to [0, 1]; missing points (0 or NaN) become -1.

    A point is missing when *both* coordinates are 0 or either is NaN — the pose model emits
    (0, 0) for undetected points, while a genuine x == 0 on a real point is not physically possible
    inside a desk ROI.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[-1] != 2:
        raise ValueError("last axis must hold (x, y)")
    out = xy / np.array([width, height], dtype=np.float64)
    missing = np.isnan(xy).any(axis=-1) | (xy == 0).all(axis=-1)
    if decimals is not None:
        out = np.round(out, decimals)
    out[missing] = MISSING
    return out


def roi_relative(xy: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    """Express pixel keypoints relative to a desk ROI ``(xmin, ymin, xmax, ymax)``, in [0, 1].

    Missing points stay -1. This was the "ROI-normalised" variant evaluated against plain
    frame normalisation for hand in pocket.
    """
    xmin, ymin, xmax, ymax = roi
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"invalid ROI {roi}")
    xy = np.asarray(xy, dtype=np.float64)
    missing = np.isnan(xy).any(axis=-1) | (xy == 0).all(axis=-1)
    out = (xy - np.array([xmin, ymin])) / np.array([xmax - xmin, ymax - ymin])
    out[missing] = MISSING
    return out


def bone_lengths(xy: np.ndarray) -> np.ndarray:
    """Euclidean length of each ``SKELETON`` bone for keypoints (..., K, 2).

    A bone with a missing endpoint gets length 0 (the convention of the distance-feature RF).
    """
    xy = np.asarray(xy, dtype=np.float64)
    missing = np.isnan(xy).any(axis=-1) | (xy == 0).all(axis=-1) | (xy == MISSING).all(axis=-1)
    a = np.array([i for i, _ in SKELETON])
    b = np.array([j for _, j in SKELETON])
    d = np.linalg.norm(xy[..., a, :] - xy[..., b, :], axis=-1)
    d[missing[..., a] | missing[..., b]] = 0.0
    return d


def kp_columns(window: int | None = None, layout: str = "kp_major") -> list[str]:
    """Column names for flattened keypoint features.

    * ``window=None`` → single frame: ``kp_0_x, kp_0_y, ..., kp_9_y``.
    * ``layout="kp_major"`` (the "regrouped" layout that worked best): all time steps of one
      coordinate are adjacent — ``kp_0_x_t0..t{W-1}, kp_0_y_t0.., kp_1_x_t0..``.
    * ``layout="time_major"``: one full frame after another — ``kp_0_x_t0, kp_0_y_t0, ...,
      kp_9_y_t0, kp_0_x_t1, ...``.
    """
    if window is None:
        return [f"kp_{k}_{c}" for k in range(NUM_KEYPOINTS) for c in ("x", "y")]
    if layout == "kp_major":
        return [
            f"kp_{k}_{c}_t{t}"
            for k in range(NUM_KEYPOINTS)
            for c in ("x", "y")
            for t in range(window)
        ]
    if layout == "time_major":
        return [
            f"kp_{k}_{c}_t{t}"
            for t in range(window)
            for k in range(NUM_KEYPOINTS)
            for c in ("x", "y")
        ]
    raise ValueError(f"unknown layout {layout!r}")
