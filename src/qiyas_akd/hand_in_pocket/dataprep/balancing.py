"""Class balancing. Positives are rare (~10 % of desk-frames), and two strategies were tried.

**Segment-window balancing** (:func:`segment_balance`, used for the deployed model) works on the
*per-frame* table, track by track (``video`` x ``desk_no``, frame-sorted):

* every run of consecutive positive frames ("segment") of length ``n`` is kept together with
  ``n // 2`` frames of context before and after it — so each positive segment is matched by about
  the same number of nearby negatives, and the model sees the *transitions* into and out of the
  behaviour, which is where false alarms happen;
* a track without any positive keeps only its first ``negative_cap`` frames (30 originally), so
  hours of "nothing happens" desks cannot swamp the data while every desk still contributes;
* optionally only a subset of videos is balanced and all others pass through unchanged (the original
  applied it to the long recordings only).

Each kept stretch gets a ``chunk`` id; :func:`windowing.make_windows` groups by it so that no window
straddles two stretches that were not adjacent in time.

Intentional deviation (bug fix): when two segments were close, the original appended overlapping
context windows separately, duplicating frames and creating windows that jumped back in time. Here
overlapping/adjacent stretches are merged before chunking, so every frame appears at most once.

**Random block balancing** (:func:`block_balance`, exploratory) works on an already *windowed*
table: draw random starting rows, take a block of ``block`` consecutive rows from the same track,
label the block by its first row, and accept it while the per-class row budget (``n_negative`` /
``n_positive``) is not exhausted. It produced fixed class ratios but random, partly redundant
coverage and lost to segment balancing on the bench. The original drew from an unseeded RNG; here
it is seeded.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd


def positive_segments(labels: Sequence[int] | np.ndarray) -> list[tuple[int, int]]:
    """``[(start, end_inclusive), ...]`` index runs of consecutive 1s."""
    lab = np.asarray(labels).astype(bool)
    if not lab.any():
        return []
    d = np.diff(np.concatenate([[0], lab.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def segment_balance(
    df: pd.DataFrame,
    label_col: str = "hand_in_pocket",
    group_cols: Sequence[str] = ("video", "desk_no"),
    negative_cap: int = 30,
    videos: Collection[str] | None = None,
) -> pd.DataFrame:
    """Keep each positive segment plus half its length of context; cap all-negative tracks.

    Returns the kept per-frame rows with an added ``chunk`` column (contiguous kept stretch id
    within the track). Every positive frame of ``df`` is guaranteed to be kept.
    """
    if negative_cap < 0:
        raise ValueError("negative_cap must be >= 0")
    parts: list[pd.DataFrame] = []
    for _, g in df.sort_values([*group_cols, "frame"]).groupby(list(group_cols), sort=True):
        g = g.reset_index(drop=True)
        if videos is not None and str(g["video"].iloc[0]) not in videos:
            parts.append(g.assign(chunk=0))
            continue
        segs = positive_segments(g[label_col].to_numpy() == 1)
        if not segs:
            parts.append(g.iloc[:negative_cap].assign(chunk=0))
            continue
        last = len(g) - 1
        halves = [(s, e, (e - s + 1) // 2) for s, e in segs]
        spans = _merge([(max(0, s - h), min(last, e + h)) for s, e, h in halves])
        for chunk, (s, e) in enumerate(spans):
            parts.append(g.iloc[s : e + 1].assign(chunk=chunk))
    if not parts:
        return df.iloc[0:0].assign(chunk=pd.Series(dtype=np.int64))
    return pd.concat(parts, ignore_index=True)


def block_balance(
    windows: pd.DataFrame,
    n_negative: int,
    n_positive: int,
    label_col: str = "hand_in_pocket",
    block: int = 5,
    group_cols: Sequence[str] = ("video", "desk_no"),
    max_attempts: int = 100_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Random block sampling of a windowed table up to a row budget per class (see module doc)."""
    if block < 1 or len(windows) < block:
        raise ValueError("block must be >= 1 and not longer than the table")
    df = windows.reset_index(drop=True)
    keys = df[list(group_cols)].astype(str).agg("|".join, axis=1).to_numpy()
    labels = df[label_col].to_numpy()
    rng = np.random.default_rng(seed)
    used: set[int] = set()
    chosen: list[np.ndarray] = []
    count = {0: 0, 1: 0}
    budget = {0: n_negative, 1: n_positive}
    attempts = 0
    while (count[0] < budget[0] or count[1] < budget[1]) and attempts < max_attempts:
        attempts += 1
        start = int(rng.integers(0, len(df) - block + 1))
        if start in used:
            continue
        used.add(start)
        if len(set(keys[start : start + block])) != 1:
            continue
        lab = int(labels[start] == 1)
        if count[lab] + block <= budget[lab]:
            chosen.append(np.arange(start, start + block))
            count[lab] += block
    if not chosen:
        return df.iloc[0:0]
    return df.iloc[np.concatenate(chosen)].reset_index(drop=True)
