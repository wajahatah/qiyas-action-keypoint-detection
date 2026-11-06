"""Per-point preprocessing shared *verbatim* by training and streaming inference.

The deployed model is sensitive to tiny numeric differences (it was trained on 3-decimal values), so
there is exactly one function, :func:`normalise_points`, that turns pixel keypoints into model
values; both the table path (:func:`normalise_table`, used for training) and the streaming path
(``infer.StreamingPredictor``) call it. Steps, in the original order:

1. **Confidence masking** — a point below ``CONF_THRESHOLD`` becomes (0, 0) = missing.
2. **Truncation** — pixel coordinates were cast to ``int`` before normalising (both in the dataset
   scripts and in the inference script), i.e. the fractional pixel is discarded.
3. **Normalisation** — one of
   ``frame``: ``x / 1280, y / 720`` (``keypoints.normalize_xy``);
   ``roi``: relative to the desk ROI (``keypoints.roi_relative``) — the ROI-normalised variant;
   ``none``: raw (truncated) pixels, missing points stay 0 — the "raw kp" variants.
4. **Missing → -1 and rounding to 3 decimals** (``frame``/``roi`` only). Rounding was a measured
   improvement ("round off" in the dataset-design study), not cosmetics.

Intentional deviation: the original table script mapped each *coordinate* that was exactly 0 to -1
independently; here a *point* is missing when both coordinates are 0 (or NaN). Undetected points are
always emitted as (0, 0), and a real point with exactly x == 0 cannot occur inside a desk ROI, so
the results are identical on real data while the rule is consistent with ``keypoints.normalize_xy``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ...desks import CameraLayout
from ...keypoints import (
    CONF_THRESHOLD,
    FRAME_H,
    FRAME_W,
    NUM_KEYPOINTS,
    ROUND_DECIMALS,
    SKELETON,
    bone_lengths,
    kp_columns,
    mask_low_confidence,
    normalize_xy,
    roi_relative,
)
from .windowing import POSITION_COLS

NORMALISE_MODES = ("frame", "roi", "none")
_KP_X = re.compile(r"^kp_(\d+)_x(_t\d+)?$")
BONE_COLS: tuple[str, ...] = tuple(f"dist_{a}_{b}" for a, b in SKELETON)


def normalise_points(
    xy: np.ndarray,
    mode: str = "frame",
    roi: tuple[float, float, float, float] | None = None,
    truncate: bool = True,
    decimals: int | None = ROUND_DECIMALS,
) -> np.ndarray:
    """Pixel keypoints (..., 2) → model values, following the steps in the module docstring."""
    xy = np.asarray(xy, dtype=np.float64)
    if truncate:
        xy = np.trunc(xy)
    if mode == "frame":
        return normalize_xy(xy, FRAME_W, FRAME_H, decimals=decimals)
    if mode == "roi":
        if roi is None:
            raise ValueError("mode 'roi' needs the desk ROI")
        out = roi_relative(xy, roi)
        return np.round(out, decimals) if decimals is not None else out
    if mode == "none":
        return np.nan_to_num(xy, nan=0.0)
    raise ValueError(f"normalise mode must be one of {NORMALISE_MODES}, got {mode!r}")


def kp_pairs(columns: Sequence[str]) -> list[tuple[str, str]]:
    """``(x_col, y_col)`` pairs of every keypoint column (per-frame or per-step) in ``columns``."""
    cols = set(columns)
    pairs = []
    for c in columns:
        m = _KP_X.match(c)
        if m:
            y = f"kp_{m.group(1)}_y{m.group(2) or ''}"
            if y not in cols:
                raise KeyError(f"{c} has no matching {y}")
            pairs.append((c, y))
    return pairs


def normalise_table(
    df: pd.DataFrame,
    mode: str = "frame",
    layout: CameraLayout | None = None,
    truncate: bool = True,
    decimals: int | None = ROUND_DECIMALS,
    desk_col: str = "desk_no",
) -> pd.DataFrame:
    """Copy of ``df`` with every keypoint column (``kp_k_x``/``kp_k_x_tN``) normalised.

    Works on per-frame and on windowed tables alike, because it only looks at column names. Mode
    ``roi`` needs a camera layout to look up each row's desk ROI.
    """
    pairs = kp_pairs(list(df.columns))
    if not pairs:
        return df.copy()
    out = df.copy()
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    xy = np.stack([df[xs].to_numpy(np.float64), df[ys].to_numpy(np.float64)], axis=-1)
    if mode == "roi":
        if layout is None:
            raise ValueError("mode 'roi' needs a CameraLayout")
        res = np.empty_like(xy)
        desks = df[desk_col].to_numpy()
        for d in np.unique(desks):
            m = desks == d
            res[m] = normalise_points(xy[m], "roi", layout.desk(int(d)).roi, truncate, decimals)
    else:
        res = normalise_points(xy, mode, None, truncate, decimals)
    out[xs] = res[..., 0]
    out[ys] = res[..., 1]
    return out


def apply_confidence_mask(df: pd.DataFrame, threshold: float = CONF_THRESHOLD) -> pd.DataFrame:
    """Set points with ``kp_k_conf < threshold`` to (0, 0) and drop the ``_conf`` columns.

    Tables without confidence columns (e.g. already-masked exports, synthetic data) pass unchanged.
    """
    conf_cols = [f"kp_{k}_conf" for k in range(NUM_KEYPOINTS)]
    if not all(c in df.columns for c in conf_cols):
        return df.drop(columns=[c for c in conf_cols if c in df.columns])
    out = df.drop(columns=conf_cols)
    masked = mask_low_confidence(frame_xy(df), df[conf_cols].to_numpy(np.float64), threshold)
    return set_frame_xy(out, masked)


def frame_xy(df: pd.DataFrame) -> np.ndarray:
    """Per-frame keypoints of a long table as an (N, K, 2) array."""
    x = df[[f"kp_{k}_x" for k in range(NUM_KEYPOINTS)]].to_numpy(np.float64)
    y = df[[f"kp_{k}_y" for k in range(NUM_KEYPOINTS)]].to_numpy(np.float64)
    return np.stack([x, y], axis=-1)


def set_frame_xy(df: pd.DataFrame, xy: np.ndarray) -> pd.DataFrame:
    """Copy of ``df`` with the per-frame keypoint columns replaced by ``xy`` (N, K, 2)."""
    out = df.copy()
    out[[f"kp_{k}_x" for k in range(NUM_KEYPOINTS)]] = xy[..., 0]
    out[[f"kp_{k}_y" for k in range(NUM_KEYPOINTS)]] = xy[..., 1]
    return out


def add_bone_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Append the 9 bone-length ("distance") features computed from pixel keypoints.

    These were the ``dist`` / ``kp+dist`` inputs of the single-frame random forests; they lost to
    raw keypoints on the bench (distances discard *where* the hand is, which is the actual signal).
    """
    out = df.copy()
    d = bone_lengths(frame_xy(df))
    for j, c in enumerate(BONE_COLS):
        out[c] = d[:, j]
    return out


def add_position_features(
    df: pd.DataFrame,
    layouts: CameraLayout | Mapping[str, CameraLayout],
    camera_col: str = "camera",
    desk_col: str = "desk_no",
) -> pd.DataFrame:
    """Write ``position_a..d`` from the desk's 4-value position descriptor in the camera layout.

    Perspective changes what "hand in pocket" looks like (a desk at the image edge is seen from the
    side), so where the desk sits in the view is a feature. With several cameras pass a mapping
    ``camera id → layout`` and a ``camera`` column. Rows whose desk is not in the layout are
    dropped, like the original script skipped rows without a matching camera/desk entry.
    """
    out = df.copy()
    if isinstance(layouts, CameraLayout):
        cams = np.full(len(df), layouts.camera_id, dtype=object)
        lookup = {layouts.camera_id: layouts}
    else:
        cams = df[camera_col].astype(str).to_numpy()
        lookup = dict(layouts)
    pos = np.full((len(df), 4), np.nan)
    for i, (cam, d) in enumerate(zip(cams, df[desk_col].to_numpy(), strict=True)):
        lay = lookup.get(cam)
        if lay is None:
            continue
        try:
            pos[i] = lay.desk(int(d)).position
        except KeyError:
            continue
    keep = ~np.isnan(pos).any(axis=1)
    for j, c in enumerate(POSITION_COLS):
        out[c] = pos[:, j]
    return out.loc[keep].reset_index(drop=True)


def frame_feature_columns(bone_features: bool = False) -> list[str]:
    """Per-frame feature columns that go into a window (20 keypoint values [+ 9 bone lengths])."""
    return kp_columns() + (list(BONE_COLS) if bone_features else [])
