"""Streaming inference for keypoint models, and per-frame / event-level evaluation on tables.

Live operation (how the deployed model ran inside the multi-camera pipeline, re-implemented here
without any pipeline code): each processed frame yields the keypoints of every person assigned to a
desk. :class:`StreamingPredictor` keeps, per desk, a buffer of the last ``window`` frames' feature
vectors (already normalised with the exact training function); as soon as a desk's buffer is full it
contributes one row — the kept offsets (e.g. t0, t2, t4), in the training column order, plus the
desk position — and **all full desks of the frame go through the model in one batched forward
pass**.

Missing detections: when a desk has no person (or a person with every keypoint missing) in a frame,
``on_missing="reset"`` (default) clears that desk's buffer, so a window never silently bridges a gap
of unknown length; ``"keep"`` leaves the buffer untouched, which is what the original offline test
script did (the deque simply waited for the next detection).

If the stream is already frame-skipped (the live system processed every 2nd frame), give the model's
own window directly: the deployed model was trained on Temp-5 windows with t1/t3 dropped, which is
the same as a 3-frame buffer on a skip-2 stream — :func:`predict_video_table` emulates the stream
with ``frame_step``.

Alerts: the original raised an alert after 5 consecutive positive predictions for a desk; that is
``evaluation.events_from_frames(pred, min_len=5)``, applied per desk by :func:`frame_events`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..desks import CameraLayout
from ..evaluation import Counts, events_from_frames, match_events
from ..keypoints import MISSING, NUM_KEYPOINTS, bone_lengths, mask_low_confidence
from .dataprep.pipeline import DataConfig
from .dataprep.preprocess import apply_confidence_mask, frame_xy, normalise_points
from .dataprep.windowing import POSITION_COLS
from .models import INPUT_KIND, TREE_KINDS, build_torch_model
from .models.sequence import to_sequences

MISSING_POLICIES = ("reset", "keep")


class KeypointPredictor:
    """A fitted keypoint model (torch tabular/sequence or sklearn-API tree) + its data recipe."""

    def __init__(
        self,
        model: Any,
        kind: str,
        data: DataConfig,
        missing_value: float = MISSING,
        device: str = "cpu",
    ) -> None:
        if kind not in TREE_KINDS and INPUT_KIND.get(kind) not in ("tabular", "sequence"):
            raise ValueError(f"model kind {kind!r} needs images; not a keypoint-only model")
        self.model = model
        self.kind = kind
        self.data = data
        self.missing_value = missing_value
        self.device = torch.device(device)
        self.feature_columns = data.feature_columns
        if isinstance(model, torch.nn.Module):
            self.model.to(self.device).eval()

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        """Positive-class probabilities for rows of ``X`` in ``feature_columns`` order."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != len(self.feature_columns):
            raise ValueError(f"expected (N, {len(self.feature_columns)}) features, got {X.shape}")
        if self.kind in TREE_KINDS:
            return self.model.predict_proba(X)[:, 1].astype(np.float64)
        if INPUT_KIND[self.kind] == "sequence":
            frame = pd.DataFrame(X, columns=self.feature_columns)
            x = to_sequences(
                frame, self.data.windowing.time_names, self.data.static_cols, self.missing_value
            )
        else:
            x = X
        with torch.no_grad():
            out = self.model(torch.from_numpy(x).to(self.device))
        return logits_to_prob(out).cpu().numpy().astype(np.float64)

    def predict_windows(self, windows: pd.DataFrame) -> np.ndarray:
        """Probabilities for a *normalised* windowed table (``dataprep.pipeline.finalise``)."""
        return self.predict_matrix(windows[self.feature_columns].to_numpy(np.float32))


def logits_to_prob(out: torch.Tensor) -> torch.Tensor:
    """(B, 1) / (B,) logits → sigmoid; (B, 2) class logits → softmax probability of class 1."""
    if out.ndim == 2 and out.shape[1] == 2:
        return torch.softmax(out, dim=1)[:, 1]
    return torch.sigmoid(out.reshape(-1))


def load_predictor(path: str | Path, device: str = "cpu") -> KeypointPredictor:
    """Rebuild a :class:`KeypointPredictor` from a ``train.py`` checkpoint (.pt or .joblib)."""
    path = Path(path)
    if path.suffix == ".joblib":
        import joblib

        ck = joblib.load(path)
        return KeypointPredictor(ck["model"], ck["kind"], DataConfig.from_dict(ck["data"]))
    ck = torch.load(path, map_location=device, weights_only=True)
    model = build_torch_model(ck["kind"], **ck["params"])
    model.load_state_dict(ck["state_dict"])
    return KeypointPredictor(
        model, ck["kind"], DataConfig.from_dict(ck["data"]), ck["missing_value"], device
    )


class StreamingPredictor:
    """Per-desk ring buffers → one batched forward pass per frame (see module docstring)."""

    def __init__(
        self,
        predictor: KeypointPredictor,
        layout: CameraLayout | None = None,
        on_missing: str = "reset",
    ) -> None:
        spec = predictor.data.windowing
        if spec.random_drop:
            raise ValueError("random frame deletion is a training-time emulation; use `keep`")
        if on_missing not in MISSING_POLICIES:
            raise ValueError(f"on_missing must be one of {MISSING_POLICIES}")
        self.predictor = predictor
        self.cfg = predictor.data
        self.layout = layout
        self.on_missing = on_missing
        self._offsets = np.array(spec.offsets)
        self._buffers: dict[int, deque[np.ndarray]] = {}

    def reset(self) -> None:
        self._buffers.clear()

    def frame_vector(self, desk_no: int, xy: np.ndarray) -> np.ndarray:
        """Per-frame features of one desk (training column order) from (K, 2) pixel keypoints."""
        cfg = self.cfg
        roi = self.layout.desk(int(desk_no)).roi if cfg.normalise == "roi" and self.layout else None
        vec = normalise_points(xy, cfg.normalise, roi, cfg.truncate_pixels, cfg.decimals)
        parts = [vec.reshape(-1)]
        if cfg.bone_features:
            parts.append(bone_lengths(xy))
        return np.concatenate(parts)

    def _position(self, desk_no: int, positions: Mapping[int, Sequence[float]] | None) -> list:
        if not self.cfg.position_features:
            return []
        if positions is not None and desk_no in positions:
            return [float(v) for v in positions[desk_no]]
        if self.layout is not None:
            return list(self.layout.desk(int(desk_no)).position)
        raise ValueError(f"no position for desk {desk_no}: pass positions or a camera layout")

    def update(
        self,
        detections: Mapping[int, np.ndarray],
        positions: Mapping[int, Sequence[float]] | None = None,
        confidences: Mapping[int, np.ndarray] | None = None,
    ) -> dict[int, float]:
        """Feed one frame: ``{desk_no: (K, 2) pixel keypoints}`` → ``{desk_no: probability}``.

        Only desks whose buffer is full get a probability.
        """
        present: dict[int, np.ndarray] = {}
        for desk, xy in detections.items():
            xy = np.asarray(xy, dtype=np.float64).reshape(NUM_KEYPOINTS, 2)
            if confidences is not None and desk in confidences:
                xy = mask_low_confidence(xy, confidences[desk], self.cfg.conf_threshold)
            if not (xy == 0).all():
                present[int(desk)] = xy
        if self.on_missing == "reset":
            for desk in [d for d in self._buffers if d not in present]:
                del self._buffers[desk]
        w = self.cfg.windowing.window
        for desk, xy in present.items():
            self._buffers.setdefault(desk, deque(maxlen=w)).append(self.frame_vector(desk, xy))
        full = [d for d, b in self._buffers.items() if len(b) == w]
        if not full:
            return {}
        kp_major = self.cfg.windowing.layout == "kp_major"
        rows = []
        for d in full:
            steps = np.stack(self._buffers[d])[self._offsets]  # (T, F)
            flat = steps.T.reshape(-1) if kp_major else steps.reshape(-1)
            rows.append(np.concatenate([flat, self._position(d, positions)]))
        probs = self.predictor.predict_matrix(np.stack(rows))
        return {d: float(p) for d, p in zip(full, probs, strict=True)}


def predict_video_table(
    frames: pd.DataFrame,
    predictor: KeypointPredictor,
    layout: CameraLayout | None = None,
    threshold: float = 0.5,
    frame_step: int = 1,
    on_missing: str = "reset",
) -> pd.DataFrame:
    """Run the streaming predictor over a per-frame table, video by video, in frame order.

    Returns one row per input ``(video, desk_no, frame)`` (after ``frame_step`` subsampling) with
    ``prob`` (NaN until the desk's buffer is full) and ``pred`` (``prob >= threshold``); the label
    column is carried over when present, ready for :func:`evaluate_events`.
    """
    label = predictor.data.label_col
    df = apply_confidence_mask(frames, predictor.data.conf_threshold)
    has_pos = all(c in df.columns for c in POSITION_COLS)
    out = []
    for video, vdf in df.sort_values(["video", "frame", "desk_no"]).groupby("video", sort=True):
        stream = StreamingPredictor(predictor, layout, on_missing)
        frame_ids = np.unique(vdf["frame"].to_numpy())[::frame_step]
        by_frame = dict(tuple(vdf[vdf["frame"].isin(frame_ids)].groupby("frame", sort=True)))
        for f in frame_ids:
            fdf = by_frame[f]
            desks = fdf["desk_no"].astype(int).to_numpy()
            xy = frame_xy(fdf)
            pos = (
                dict(zip(desks, fdf[list(POSITION_COLS)].to_numpy(), strict=True))
                if has_pos
                else None
            )
            probs = stream.update(dict(zip(desks, xy, strict=True)), pos)
            for i, d in enumerate(desks):
                row = {"video": video, "desk_no": d, "frame": f, "prob": probs.get(d, np.nan)}
                if label in fdf.columns:
                    row[label] = int(fdf[label].iloc[i])
                out.append(row)
    res = pd.DataFrame(out)
    if res.empty:
        return pd.DataFrame(columns=["video", "desk_no", "frame", "prob", "pred"])
    res["pred"] = (res["prob"].fillna(-np.inf) >= threshold).astype(int)
    return res


def frame_events(
    pred: pd.DataFrame, col: str = "pred", min_len: int = 1
) -> dict[tuple[str, int], list[tuple[int, int]]]:
    """Per ``(video, desk_no)`` track: ``(first_frame, last_frame)`` of each run of ``col`` == 1."""
    events: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for (video, desk), g in pred.sort_values("frame").groupby(["video", "desk_no"], sort=True):
        frames = g["frame"].to_numpy()
        runs = events_from_frames(g[col].to_numpy(), min_len=min_len)
        events[(str(video), int(desk))] = [(int(frames[s]), int(frames[e])) for s, e in runs]
    return events


def evaluate_events(
    pred: pd.DataFrame,
    label_col: str = "hand_in_pocket",
    min_len: int = 1,
    tolerance: int = 0,
) -> Counts:
    """Event-level TP/FP/FN summed over all desk tracks (predictions persistence-filtered)."""
    predicted = frame_events(pred, "pred", min_len)
    truth = frame_events(pred, label_col, 1)
    total = Counts(0, 0, 0)
    for key, t in truth.items():
        total = total + match_events(predicted.get(key, []), t, tolerance)
    return total
