"""The dataset recipe as one config object, applied identically for training and inference.

The original pipeline was ~10 scripts run by hand in sequence (annotate → window → position values →
regroup + normalise → balance → augment → skip-rate emulation → train), each with its own hardcoded
paths, so a dataset version was only reproducible from memory. Here the whole recipe is a
:class:`DataConfig` (one YAML block), and the order of operations is fixed in code:

1. :func:`prepare_frames` — confidence masking, optional bone lengths, desk position features;
2. :func:`build_windows` — optional segment balancing (per-frame), windowing + labelling, optional
   block balancing (per-window). Values are still **pixels** here;
3. ``augment.jitter`` on the *training split only* (pixel space);
4. :func:`finalise` — normalisation + rounding of all keypoint columns.

Splitting happens between 2 and 3 (by video, ``evaluation.grouped_split``), so augmented copies
never leak into validation and no video is on both sides.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from ...desks import CameraLayout
from ...keypoints import CONF_THRESHOLD, ROUND_DECIMALS
from .balancing import block_balance, segment_balance
from .preprocess import (
    NORMALISE_MODES,
    add_bone_lengths,
    add_position_features,
    apply_confidence_mask,
    frame_feature_columns,
    normalise_table,
)
from .windowing import POSITION_COLS, WindowSpec, make_windows, windowed_feature_columns

BALANCE_METHODS = ("none", "segment", "block")


@dataclass(frozen=True)
class BalanceConfig:
    """``method``: none | segment (per-frame, deployed) | block (per-window, exploratory)."""

    method: str = "none"
    negative_cap: int = 30
    n_negative: int = 30_000
    n_positive: int = 6_000
    block: int = 5
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method not in BALANCE_METHODS:
            raise ValueError(f"balance method must be one of {BALANCE_METHODS}")


@dataclass(frozen=True)
class AugmentConfig:
    """Jitter augmentation (see ``augment.jitter``); both copy counts 0 disables it."""

    max_px: int = 4
    times_positive: int = 0
    times_negative: int = 0
    per_value: bool = True
    seed: int = 0

    @property
    def enabled(self) -> bool:
        return self.times_positive > 0 or self.times_negative > 0


@dataclass(frozen=True)
class DataConfig:
    """Everything that defines a dataset version (YAML block ``data:``)."""

    label_col: str = "hand_in_pocket"
    conf_threshold: float = CONF_THRESHOLD
    normalise: str = "frame"
    truncate_pixels: bool = True
    decimals: int | None = ROUND_DECIMALS
    bone_features: bool = False
    position_features: bool = True
    windowing: WindowSpec = field(default_factory=WindowSpec)
    balance: BalanceConfig = field(default_factory=BalanceConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)

    def __post_init__(self) -> None:
        if self.normalise not in NORMALISE_MODES:
            raise ValueError(f"normalise must be one of {NORMALISE_MODES}")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> DataConfig:
        d = dict(d or {})
        win = d.pop("windowing", {}) or {}
        if win.get("keep") is not None:
            win["keep"] = tuple(win["keep"])
        return cls(
            windowing=WindowSpec(**win),
            balance=BalanceConfig(**(d.pop("balance", {}) or {})),
            augment=AugmentConfig(**(d.pop("augment", {}) or {})),
            **d,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def frame_features(self) -> list[str]:
        """Per-frame columns that are windowed."""
        return frame_feature_columns(self.bone_features)

    @property
    def static_cols(self) -> list[str]:
        return list(POSITION_COLS) if self.position_features else []

    @property
    def feature_columns(self) -> list[str]:
        """Model input columns, in model order: windowed features, then desk position."""
        return windowed_feature_columns(self.windowing, self.frame_features) + self.static_cols


def prepare_frames(
    frames: pd.DataFrame, cfg: DataConfig, layout: CameraLayout | None = None
) -> pd.DataFrame:
    """Per-frame cleaning: confidence mask, bone lengths, position features (pixels kept)."""
    df = apply_confidence_mask(frames, cfg.conf_threshold)
    if cfg.bone_features:
        df = add_bone_lengths(df)
    if cfg.position_features and not all(c in df.columns for c in POSITION_COLS):
        if layout is None:
            raise ValueError("position features need position_* columns or a camera layout")
        df = add_position_features(df, layout)
    return df


def build_windows(frames: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Balanced, labelled windows in pixel space from a prepared per-frame table."""
    df = frames
    if cfg.balance.method == "segment":
        df = segment_balance(df, cfg.label_col, negative_cap=cfg.balance.negative_cap)
    win = make_windows(
        df,
        cfg.windowing,
        label_col=cfg.label_col,
        frame_features=cfg.frame_features,
        static_cols=cfg.static_cols,
    )
    if cfg.balance.method == "block":
        b = cfg.balance
        win = block_balance(win, b.n_negative, b.n_positive, cfg.label_col, b.block, seed=b.seed)
    return win


def finalise(
    windows: pd.DataFrame, cfg: DataConfig, layout: CameraLayout | None = None
) -> pd.DataFrame:
    """Normalise + round every keypoint column (the last step before the model)."""
    return normalise_table(windows, cfg.normalise, layout, cfg.truncate_pixels, cfg.decimals)
