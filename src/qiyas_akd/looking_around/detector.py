"""Looking-around decision: per-person geometry rule → per-desk persistence filter → alert flags.

Pipeline per desk (one track of one candidate's keypoints over time):

1. **Single-frame decision** (``decide``) with one of three methods

   * ``ear_line`` — ``abs(ear_line_angle) > threshold`` (the method that won the offline
     comparison);
   * ``centre_angle`` — ``abs(centre_angle) > threshold``;
   * ``triangle`` — head outside the visual-field cone whose half-angle is the threshold.

   The threshold is any ``thresholds.Threshold`` callable (linear depth-scaled by default).
   A frame where a required keypoint is missing yields ``None`` (no decision): no angle is computed
   when an ear is missing, because a ``(0, 0)`` ear produces a spurious extreme angle.
2. **Persistence filter** (``persistence_filter``) — a single noisy pose frame must never raise an
   alert, so a frame is flagged only when the behaviour held in at least ``min_hits`` of the last
   ``window`` frames (and, by default, in the current frame too).

There is no learned model for looking around: thresholds are calibrated per camera (see
``desks.Desk``) and every decision is explainable as an angle against a threshold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..desks import CameraLayout, Desk
from ..evaluation import Counts, events_from_frames, match_events
from ..keypoints import FRAME_H, FRAME_W, KP, NUM_KEYPOINTS
from .geometry import (
    DEFAULT_CONE_LENGTH,
    centre_angle,
    ear_line_angle,
    head_in_visual_field,
    is_missing,
)
from .thresholds import LinearThreshold, Threshold, make_threshold

METHODS = ("ear_line", "centre_angle", "triangle")
LABEL_COLUMN = "looking_around"


@dataclass(frozen=True)
class LookingAroundConfig:
    """Everything that defines one looking-around variant.

    ``window`` / ``min_hits`` / ``require_current`` parameterise the persistence filter; the
    defaults (4 of the last 5 frames, current frame included) are a generic choice — the deployed
    system used a 5-frame window with a mean above 0.8, i.e. ``min_hits=5``.
    """

    method: str = "ear_line"
    threshold: Threshold = field(default_factory=LinearThreshold)
    cone_length: float = DEFAULT_CONE_LENGTH
    window: int = 5
    min_hits: int = 4
    require_current: bool = True

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if not 1 <= self.min_hits <= self.window:
            raise ValueError("need 1 <= min_hits <= window")

    @classmethod
    def from_dict(cls, d: Mapping) -> LookingAroundConfig:
        """Build from a YAML-style dict; ``threshold`` is a ``thresholds.make_threshold`` spec."""
        d = dict(d)
        known = {"method", "threshold", "cone_length", "window", "min_hits", "require_current"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown looking-around config keys: {sorted(unknown)}")
        d["threshold"] = make_threshold(d.get("threshold"))
        return cls(**d)


def decide(
    xy: np.ndarray,
    desk: Desk,
    method: str = "ear_line",
    threshold: Threshold | None = None,
    cone_length: float = DEFAULT_CONE_LENGTH,
) -> bool | None:
    """Single-frame looking-around decision for one person's keypoints ``(K, 2)`` in pixels.

    Returns ``True`` / ``False``, or ``None`` when a keypoint the method needs is missing.
    Strictly greater than the threshold counts as looking around (the original left a frame at
    exactly the threshold unchanged).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    threshold = LinearThreshold() if threshold is None else threshold
    xy = np.asarray(xy, dtype=float)
    if xy.shape != (NUM_KEYPOINTS, 2):
        raise ValueError(f"expected ({NUM_KEYPOINTS}, 2) keypoints, got {xy.shape}")
    head, left, right = xy[KP["head"]], xy[KP["left_ear"]], xy[KP["right_ear"]]
    if is_missing(left) or is_missing(right):
        return None

    if method == "ear_line":
        angle = ear_line_angle(left, right)
        return None if angle is None else abs(angle) > threshold(left, right, desk)
    if method == "centre_angle":
        angle = centre_angle(head, left, right)
        return None if angle is None else abs(angle) > threshold(left, right, desk)
    inside = head_in_visual_field(head, left, right, threshold(left, right, desk), cone_length)
    return None if inside is None else not inside


def persistence_filter(
    raw: Sequence[int | bool] | np.ndarray,
    window: int = 5,
    min_hits: int = 4,
    require_current: bool = True,
) -> np.ndarray:
    """Flag frame ``t`` when ``raw`` was positive in ≥ ``min_hits`` of frames ``t−window+1 … t``.

    With ``require_current`` the current frame must be positive as well, so an alert ends as soon
    as the behaviour stops instead of lingering for up to ``window`` frames. Returns a 0/1 int
    array. The first ``min_hits − 1`` frames of every event are not flagged — the price of
    suppressing blips; event-level matching by overlap is unaffected.
    """
    if not 1 <= min_hits <= window:
        raise ValueError("need 1 <= min_hits <= window")
    r = np.asarray(raw).astype(bool)
    if r.ndim != 1:
        raise ValueError("raw must be 1-D")
    csum = np.concatenate([[0], np.cumsum(r, dtype=np.int64)])
    idx = np.arange(len(r))
    hits = csum[idx + 1] - csum[np.maximum(idx + 1 - window, 0)]
    out = hits >= min_hits
    if require_current:
        out &= r
    return out.astype(int)


def detect_sequence(xy_seq: np.ndarray, desk: Desk, config: LookingAroundConfig) -> np.ndarray:
    """Per-frame alert flags (0/1) for one desk's keypoint track ``(T, K, 2)``.

    Frames without a decision (missing keypoints) count as negative.
    """
    raw = raw_decisions(xy_seq, desk, config)
    return persistence_filter(raw, config.window, config.min_hits, config.require_current)


def raw_decisions(xy_seq: np.ndarray, desk: Desk, config: LookingAroundConfig) -> np.ndarray:
    """Unfiltered single-frame decisions (0/1; ``None`` mapped to 0) for a track ``(T, K, 2)``."""
    xy_seq = np.asarray(xy_seq, dtype=float)
    if xy_seq.ndim != 3:
        raise ValueError("xy_seq must have shape (T, K, 2)")
    return np.array(
        [
            bool(decide(f, desk, config.method, config.threshold, config.cone_length))
            for f in xy_seq
        ],
        dtype=int,
    )


def _desk_lookup(desks: CameraLayout | Mapping[int, Desk]) -> Mapping[int, Desk]:
    if isinstance(desks, CameraLayout):
        return {d.desk_id: d for d in desks.desks}
    return desks


def _track(group: pd.DataFrame) -> np.ndarray:
    cols = [f"kp_{k}_{c}" for k in range(NUM_KEYPOINTS) for c in ("x", "y")]
    return group[cols].to_numpy(dtype=float).reshape(len(group), NUM_KEYPOINTS, 2)


def detect_table(
    df: pd.DataFrame,
    desks: CameraLayout | Mapping[int, Desk],
    config: LookingAroundConfig | None = None,
) -> pd.DataFrame:
    """Run the detector over a frame table (``video, frame, desk_no, kp_{k}_x, kp_{k}_y`` pixels).

    Returns a copy sorted by ``video, desk_no, frame`` with two extra columns: ``la_raw`` (single
    frame decision) and ``la_flag`` (after the persistence filter, which runs per video and desk).
    """
    config = LookingAroundConfig() if config is None else config
    lookup = _desk_lookup(desks)
    out = df.sort_values(["video", "desk_no", "frame"]).reset_index(drop=True)
    raw = np.zeros(len(out), dtype=int)
    flag = np.zeros(len(out), dtype=int)
    for (_, desk_no), group in out.groupby(["video", "desk_no"], sort=False):
        if int(desk_no) not in lookup:
            raise KeyError(f"no desk layout for desk_no={desk_no}")
        desk = lookup[int(desk_no)]
        r = raw_decisions(_track(group), desk, config)
        raw[group.index] = r
        flag[group.index] = persistence_filter(
            r, config.window, config.min_hits, config.require_current
        )
    out["la_raw"] = raw
    out["la_flag"] = flag
    return out


def evaluate_table(
    df: pd.DataFrame,
    desks: CameraLayout | Mapping[int, Desk],
    config: LookingAroundConfig | None = None,
    label_column: str = LABEL_COLUMN,
    tolerance: int = 0,
) -> Counts:
    """Event-level TP / FP / FN of the detector against a labelled frame table.

    Predicted and true events are built per (video, desk) with ``events_from_frames`` and matched
    with ``match_events`` — alerts, not frames, are what an invigilator acts on.
    """
    if label_column not in df.columns:
        raise KeyError(f"label column {label_column!r} not in table")
    pred = detect_table(df, desks, config)
    total = Counts(0, 0, 0)
    for _, group in pred.groupby(["video", "desk_no"], sort=False):
        total = total + match_events(
            events_from_frames(group["la_flag"].to_numpy()),
            events_from_frames(group[label_column].to_numpy()),
            tolerance=tolerance,
        )
    return total


def synthetic_layout(n_desks: int = 4, camera_id: str = "synthetic") -> CameraLayout:
    """Desk layout matching ``synthetic.generate_frame_table`` (desk centres evenly spaced from
    x = 160 to x = W − 160, heads near 0.45·H).

    The reference line sits above the synthetic heads, so the ears are "inside" the band and
    ``base_angle_in`` applies. The base angles (45° in, 60° out) are a calibration for the synthetic
    kinematics — whose looking-around events rotate the ear line by 50–80° — not deployment values.
    """
    if n_desks < 1:
        raise ValueError("n_desks must be >= 1")
    centres = np.linspace(160, FRAME_W - 160, n_desks)
    half = 160.0 if n_desks == 1 else min(160.0, (centres[1] - centres[0]) / 2.0)
    ref_y = 0.35 * FRAME_H
    desks = tuple(
        Desk(
            desk_id=i + 1,
            roi=(float(c - half), 0.2 * FRAME_H, float(c + half), float(FRAME_H)),
            ref_line=(float(c - half), ref_y, float(c + half), ref_y),
            base_angle_in=45.0,
            base_angle_out=60.0,
        )
        for i, c in enumerate(centres)
    )
    return CameraLayout(camera_id=camera_id, desks=desks)
