"""Evaluation utilities: counts → metrics, per-frame predictions → events, grouped splits.

Why event-level: an exam invigilator acts on *alerts*, not frames. Every model in this repo was
therefore judged on video test benches by counting alert events as TP / FP / FN, not by per-frame
accuracy (which is dominated by the ~90 % negative frames and by near-duplicate consecutive frames).

Why grouped splits: consecutive frames of the same video are near-identical. A random frame-level
split puts near-copies in train and validation and inflates validation scores (this is what made
several val-split numbers in the project look near-perfect). ``grouped_split`` keeps every video in
exactly one side.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Counts:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def events_from_frames(pred: Sequence[int] | np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """Turn a per-frame 0/1 sequence into events ``[(start, end_inclusive), ...]``.

    Runs shorter than ``min_len`` frames are discarded (a persistence filter: a behaviour must hold
    for several consecutive frames before it becomes an alert).
    """
    p = np.asarray(pred).astype(bool)
    if p.ndim != 1:
        raise ValueError("pred must be 1-D")
    events: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(p):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                events.append((start, i - 1))
            start = None
    if start is not None and len(p) - start >= min_len:
        events.append((start, len(p) - 1))
    return events


def match_events(
    predicted: Sequence[tuple[int, int]],
    truth: Sequence[tuple[int, int]],
    tolerance: int = 0,
) -> Counts:
    """Greedy one-to-one matching of predicted to ground-truth events by temporal overlap.

    A prediction is a TP if it overlaps (within ``tolerance`` frames) a not-yet-matched truth
    event; unmatched predictions are FP; unmatched truth events are FN.
    """
    matched: set[int] = set()
    tp = fp = 0
    for ps, pe in predicted:
        hit = None
        for j, (ts, te) in enumerate(truth):
            if j not in matched and ps <= te + tolerance and pe >= ts - tolerance:
                hit = j
                break
        if hit is None:
            fp += 1
        else:
            matched.add(hit)
            tp += 1
    return Counts(tp=tp, fp=fp, fn=len(truth) - len(matched))


def grouped_split(
    groups: Sequence[str] | np.ndarray, test_size: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices so that no group (e.g. video id) appears on both sides.

    Returns ``(train_idx, test_idx)``. Deterministic for a given ``seed``.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise ValueError("need at least two groups for a grouped split")
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * test_size)))
    test_groups = set(uniq[:n_test].tolist())
    is_test = np.array([g in test_groups for g in groups])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)
