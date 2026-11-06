"""±N-pixel jitter augmentation of keypoint rows.

Semantics preserved from the original augmentation script (the "+ ±4 px jitter" rows of the bench,
which gave the best 5-frame MLP, F1 0.89):

* applied in **pixel space, before normalisation** — so the jitter is a true ±4 px (≈ 0.003 in
  normalised x), comparable to pose-model noise;
* every non-missing coordinate is truncated to an integer and shifted by an independent integer
  drawn uniformly from ``[-max_px, max_px]`` (``per_value=False`` uses one shift per row instead);
* a missing coordinate (0) stays 0 — jitter must never invent a keypoint the pose model did not see;
* each source row is kept (``original = 1``) and jittered copies are added (``original = 0``):
  ``times_positive`` per positive row and ``times_negative`` per negative row — the study used e.g.
  "TP x5, FN x2" to also rebalance classes; ``times_negative = 0`` augments positives only.

Differences, all intentional:

* it is applied to **windowed** rows (all ``kp_k_{x|y}_tN`` columns), not to per-frame rows. The
  original interleaved jittered copies with the original frames of a per-frame file, so a later
  sliding window mixed real and jittered frames; per-window jitter has no such artefact;
* a jittered valid coordinate is clipped to ``[1, frame size - 1]`` — without the clip a point near
  the border could be shifted to exactly 0 and silently turn into a *missing* point;
* the original script augmented every row of whatever file it was given the same number of times;
  per-class copy counts express the same thing for a mixed table.

Train-only: the pipeline augments after the video-grouped split, so validation rows are never
copies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...keypoints import FRAME_H, FRAME_W
from .preprocess import kp_pairs


def jitter(
    df: pd.DataFrame,
    max_px: int = 4,
    times_positive: int = 1,
    times_negative: int = 0,
    label_col: str = "hand_in_pocket",
    per_value: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Return ``df`` (``original = 1``) plus jittered copies (``original = 0``) per class."""
    if max_px < 0 or times_positive < 0 or times_negative < 0:
        raise ValueError("max_px and the copy counts must be >= 0")
    pairs = kp_pairs(list(df.columns))
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    base = df.copy()
    base["original"] = 1
    if not pairs:
        return base.reset_index(drop=True)
    pos = (df[label_col] == 1).to_numpy()
    sources = [df[pos]] * times_positive + [df[~pos]] * times_negative
    rng = np.random.default_rng(seed)
    copies = []
    for src in sources:
        if src.empty:
            continue
        row_noise = rng.integers(-max_px, max_px + 1, size=(len(src), 1))
        moved_cols: dict[str, np.ndarray] = {}
        for cols, limit in ((xs, FRAME_W), (ys, FRAME_H)):
            v = src[cols].to_numpy(np.float64)
            noise = rng.integers(-max_px, max_px + 1, size=v.shape) if per_value else row_noise
            moved = np.where(v != 0, np.clip(np.trunc(v) + noise, 1, limit - 1), 0.0)
            moved_cols.update(zip(cols, moved.T, strict=True))
        new = pd.DataFrame(moved_cols, index=src.index)
        aug = pd.concat([src.drop(columns=xs + ys), new], axis=1)[list(src.columns)]
        copies.append(aug.assign(original=0))
    return pd.concat([base, *copies], ignore_index=True)
