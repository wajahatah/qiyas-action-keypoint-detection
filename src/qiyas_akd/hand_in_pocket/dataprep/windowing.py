"""Sliding temporal windows over per-desk keypoint tracks, plus the studied labelling rules.

Input is the long per-frame table (one row per ``video`` x ``frame`` x ``desk_no``, see
``qiyas_akd.synthetic.generate_frame_table``). For every ``(video, desk_no)`` track, sorted by
frame, windows of ``window`` consecutive rows are taken with step ``stride``; each window becomes
one row.

Vocabulary of the dataset-design study (kept identical so results tables stay readable):

* **Temp-N** — ``window = N`` consecutive frames per row.
* **Frame skipping (fs)** — ``keep=(0, 2, 4)`` drops t1/t3 from a Temp-5 window, so a window still
  spans 5 frames of time but carries only 3 frames of features. This is what made the deployed model
  match the live stream, which delivers every 2nd frame: 3 received frames == t0, t2, t4.
* **Row skipping (rs)** — ``row_step = 2`` deletes every 2nd *window* of a track (the temporal
  columns stay intact), mimicking the capture-rate drop of the stream.
* **Frame step** — ``frame_step = 2`` deletes every 2nd *frame* before windowing (the older
  per-frame variant of skip-rate emulation).
* **Random frame deletion (Dataset D / "t4")** — ``random_drop = 1`` removes one random frame per
  window (5 → 4 frames) to mimic irregular drops.
* **Labelling rules** — the label of a window is derived from its per-frame labels:
  ``any`` (Any-One: any frame is 1 — the rule of the deployed model), ``all`` (All-One),
  ``fraction`` (e.g. 0.5 = the 50 % rule), ``min_count`` (e.g. >= 3 of 5), ``center`` (the label of
  the middle frame, used by the crop/hybrid data loaders).
* **Transition removal ("ws", without sequence)** — ``drop_mixed=True`` drops windows that contain
  both 0 and 1 frames, keeping only unambiguous windows.

Labels are always computed on the *full* window (all ``window`` frames), not only the kept ones —
the original pipeline labelled the 5-frame window first and dropped t1/t3 afterwards.

Output columns: ``video, desk_no[, camera][, chunk], frame (first frame), frame_end (last frame),
frame_<tn> per kept step, <features>_<tn>, <static columns>, label_fraction, <label>``. Feature
names use the *original* offset (``t0, t2, t4`` for the 3-frame model), which keeps the deployed
model's column names; with ``random_drop`` the dropped offset differs per row, so steps are
renumbered ``t0..t{n-1}``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...keypoints import kp_columns

LABEL_RULES = ("any", "all", "fraction", "min_count", "center")
POSITION_COLS: tuple[str, ...] = ("position_a", "position_b", "position_c", "position_d")


@dataclass(frozen=True)
class WindowSpec:
    """How a per-frame track is cut into labelled windows (all fields map 1:1 to YAML)."""

    window: int = 5
    stride: int = 1
    keep: tuple[int, ...] | None = None
    label_rule: str = "any"
    label_param: float | None = None
    drop_mixed: bool = False
    row_step: int = 1
    frame_step: int = 1
    random_drop: int = 0
    layout: str = "kp_major"
    max_frame_gap: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.window < 1 or self.stride < 1 or self.row_step < 1 or self.frame_step < 1:
            raise ValueError("window, stride, row_step and frame_step must be >= 1")
        if self.keep is not None:
            object.__setattr__(self, "keep", tuple(int(k) for k in self.keep))
            if not self.keep or sorted(set(self.keep)) != list(self.keep):
                raise ValueError(f"keep must be strictly increasing offsets, got {self.keep}")
            if self.keep[0] < 0 or self.keep[-1] >= self.window:
                raise ValueError(f"keep offsets {self.keep} outside window {self.window}")
        if not 0 <= self.random_drop < self.window:
            raise ValueError("random_drop must be in [0, window)")
        if self.random_drop and self.keep is not None:
            raise ValueError("random_drop and keep are mutually exclusive")
        if self.label_rule not in LABEL_RULES:
            raise ValueError(f"label_rule must be one of {LABEL_RULES}, got {self.label_rule!r}")
        if self.label_rule in ("fraction", "min_count") and self.label_param is None:
            raise ValueError(f"label_rule {self.label_rule!r} needs label_param")
        if self.layout not in ("kp_major", "time_major"):
            raise ValueError(f"unknown layout {self.layout!r}")

    @property
    def offsets(self) -> tuple[int, ...]:
        """Offsets inside the window whose features are kept (ignores ``random_drop``)."""
        return tuple(range(self.window)) if self.keep is None else self.keep

    @property
    def n_steps(self) -> int:
        """Number of time steps per output row."""
        return self.window - self.random_drop if self.random_drop else len(self.offsets)

    @property
    def time_names(self) -> list[str]:
        """Suffixes of the per-step columns, e.g. ``['t0', 't2', 't4']``."""
        if self.random_drop:
            return [f"t{i}" for i in range(self.n_steps)]
        return [f"t{o}" for o in self.offsets]


def window_label(labels: np.ndarray, rule: str, param: float | None = None) -> np.ndarray:
    """Window labels (N,) from per-frame labels (N, W) according to a labelling rule."""
    lab = np.asarray(labels)
    if lab.ndim != 2:
        raise ValueError("labels must be (N, W)")
    pos = lab == 1
    if rule == "any":
        out = pos.any(axis=1)
    elif rule == "all":
        out = pos.all(axis=1)
    elif rule == "fraction":
        out = pos.mean(axis=1) >= float(param if param is not None else 0.5)
    elif rule == "min_count":
        out = pos.sum(axis=1) >= int(param if param is not None else 1)
    elif rule == "center":
        out = pos[:, lab.shape[1] // 2]
    else:
        raise ValueError(f"unknown label rule {rule!r}")
    return out.astype(np.int64)


def windowed_feature_columns(
    spec: WindowSpec, frame_features: Sequence[str] | None = None
) -> list[str]:
    """Names of the per-step feature columns in the order the model sees them.

    ``kp_major`` puts all steps of one feature next to each other (``kp_0_x_t0, kp_0_x_t2, ...``) —
    the "regrouped" layout that beat ``time_major`` for every model family on the test bench.
    """
    feats = list(kp_columns() if frame_features is None else frame_features)
    names = spec.time_names
    if spec.layout == "kp_major":
        return [f"{f}_{t}" for f in feats for t in names]
    return [f"{f}_{t}" for t in names for f in feats]


def _runs(frames: np.ndarray, max_gap: int | None) -> list[np.ndarray]:
    """Index runs of a frame-sorted track, split where the frame gap exceeds ``max_gap``."""
    idx = np.arange(len(frames))
    if max_gap is None or len(frames) < 2:
        return [idx]
    cuts = np.flatnonzero(np.diff(frames) > max_gap) + 1
    return np.split(idx, cuts)


def make_windows(
    df: pd.DataFrame,
    spec: WindowSpec,
    label_col: str = "hand_in_pocket",
    frame_features: Sequence[str] | None = None,
    static_cols: Sequence[str] | None = None,
    group_cols: Sequence[str] = ("video", "desk_no"),
) -> pd.DataFrame:
    """Cut every track of a per-frame table into labelled windows (see module docstring).

    ``static_cols`` (default: the desk position columns present in ``df``) are copied from the first
    frame of each window. If ``df`` has a ``chunk`` column (written by segment balancing) it is
    added to the grouping, so a window never straddles two kept chunks. Tracks shorter than
    ``window`` are skipped, as in the original scripts. Deterministic for a given ``spec.seed``.
    """
    feats = list(kp_columns() if frame_features is None else frame_features)
    statics = [c for c in (POSITION_COLS if static_cols is None else static_cols) if c in df]
    groups = list(group_cols)
    if "chunk" in df.columns and "chunk" not in groups:
        groups.append("chunk")
    meta_extra = [c for c in ("camera",) if c in df.columns and c not in groups]
    missing = [c for c in [*groups, "frame", label_col, *feats] if c not in df.columns]
    if missing:
        raise KeyError(f"frame table lacks columns {missing}")

    rng = np.random.default_rng(spec.seed)
    w, names = spec.window, spec.time_names
    feat_cols = windowed_feature_columns(spec, feats)
    blocks: list[pd.DataFrame] = []
    for key, g in df.sort_values([*groups, "frame"]).groupby(groups, sort=True):
        g = g.iloc[:: spec.frame_step]
        frames_all = g["frame"].to_numpy()
        for run in _runs(frames_all, spec.max_frame_gap):
            if len(run) < w:
                continue
            sub = g.iloc[run]
            starts = np.arange(0, len(sub) - w + 1, spec.stride)[:: spec.row_step]
            idx = starts[:, None] + np.arange(w)
            frame_labels = sub[label_col].to_numpy()[idx]
            frac = (frame_labels == 1).mean(axis=1)
            keep_rows = ~((frac > 0) & (frac < 1)) if spec.drop_mixed else np.ones(len(idx), bool)
            if not keep_rows.any():
                continue
            idx, frame_labels, frac = idx[keep_rows], frame_labels[keep_rows], frac[keep_rows]
            if spec.random_drop:
                sel = np.sort(
                    np.stack([rng.permutation(w)[: spec.n_steps] for _ in range(len(idx))]), axis=1
                )
            else:
                sel = np.broadcast_to(np.array(spec.offsets), (len(idx), spec.n_steps))
            step_idx = np.take_along_axis(idx, sel, axis=1)  # (N, T) row indices into sub
            values = sub[feats].to_numpy(dtype=np.float64)[step_idx]  # (N, T, F)
            if spec.layout == "kp_major":
                flat = values.transpose(0, 2, 1).reshape(len(idx), -1)
            else:
                flat = values.reshape(len(idx), -1)
            frames = sub["frame"].to_numpy()
            out = pd.DataFrame(flat, columns=feat_cols)
            key_t = key if isinstance(key, tuple) else (key,)
            meta = {c: v for c, v in zip(groups, key_t, strict=True)}
            for c in meta_extra:
                meta[c] = sub[c].iloc[0]
            meta["frame"] = frames[idx[:, 0]]
            meta["frame_end"] = frames[idx[:, -1]]
            for j, t in enumerate(names):
                meta[f"frame_{t}"] = frames[step_idx[:, j]]
            head = pd.DataFrame(meta, index=out.index)
            tail = pd.DataFrame({c: sub[c].to_numpy()[idx[:, 0]] for c in statics}, index=out.index)
            tail["label_fraction"] = frac
            tail[label_col] = window_label(frame_labels, spec.label_rule, spec.label_param)
            blocks.append(pd.concat([head, out, tail], axis=1))
    if not blocks:
        cols = [*groups, *meta_extra, "frame", "frame_end", *[f"frame_{t}" for t in names]]
        return pd.DataFrame(columns=[*cols, *feat_cols, *statics, "label_fraction", label_col])
    return pd.concat(blocks, ignore_index=True)
