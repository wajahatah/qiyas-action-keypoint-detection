"""Config/CLI training for the hand-raise keypoint models (two-branch MLP, MLP, RF) and the rule.

Usage::

    python -m qiyas_akd.hand_raise.train --synthetic --model two_branch --epochs 5
    python -m qiyas_akd.hand_raise.train --data a.csv b.csv --config two_branch.yaml

Settings resolve as defaults < ``--config`` YAML < explicit CLI flags.

Evaluation protocol (and why):

* **Split by video** (``evaluation.grouped_split``). Consecutive frames are near-duplicates; a
  random frame split leaks them across train and validation. The original MLP/RF runs that used
  held-out *videos* gave the honest numbers (RF hand-raise recall 0.43); frame-level splits
  elsewhere in the project produced inflated, near-perfect scores.
* **Early stopping on an inner split of the training videos**, never on the reported validation
  videos — the original monitored the same validation set it reported, which biases the result.
* **Frame- and event-level metrics.** Hand raise is rare (~22 % of frames); frame accuracy mostly
  measures the negative class, so precision / recall / F1 are reported, plus TP/FP/FN of alert
  events per (video, desk).
* **Decision threshold** 0.6 on the sigmoid output for the MLPs (as in the original two-input
  inference script) and 0.5 (``predict``) for the random forest, unless overridden.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..desks import CameraLayout
from ..evaluation import Counts, events_from_frames, grouped_split, match_events
from ..synthetic import generate_frame_table
from .data import LABEL_COLUMN, load_tables, validate_binary_labels
from .features import VARIANTS, build_features, roi_left_from_layout
from .models import KeypointMLP, TwoBranchMLP, make_random_forest
from .rules import predict_table

MODELS = ("two_branch", "mlp", "rf", "rule")
_DEFAULT_THRESHOLD = {"two_branch": 0.6, "mlp": 0.6, "rf": 0.5, "rule": 0.5}


@dataclass
class TrainConfig:
    """One training run. Field names are the YAML keys."""

    model: str = "two_branch"
    variant: str = "normalise"
    engineered: bool = False
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    patience: int = 10
    test_size: float = 0.2
    seed: int = 42
    threshold: float | None = None
    label_column: str = LABEL_COLUMN
    group_column: str = "video"
    track_column: str = "desk_no"
    n_estimators: int = 10
    rf_params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {self.model!r}")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        if self.model == "two_branch" and self.engineered:
            raise ValueError("two_branch takes exactly the [x block, y block] features; "
                             "use model 'mlp' or 'rf' with engineered features")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")

    @property
    def decision_threshold(self) -> float:
        return _DEFAULT_THRESHOLD[self.model] if self.threshold is None else float(self.threshold)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)


@dataclass
class TrainResult:
    config: TrainConfig
    model: object
    metrics: dict


def _seed_everything(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def _build_torch_model(config: TrainConfig, input_dim: int) -> nn.Module:
    if config.model == "two_branch":
        return TwoBranchMLP()
    return KeypointMLP(input_dim=input_dim)


def _predict_proba_torch(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).squeeze(-1).numpy()


def _fit_torch(
    config: TrainConfig,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[nn.Module, dict]:
    gen = _seed_everything(config.seed)
    model = _build_torch_model(config, X.shape[1])
    early_stop = config.patience > 0 and len(np.unique(groups)) >= 2
    if early_stop:
        fit_idx, stop_idx = grouped_split(groups, config.test_size, seed=config.seed + 1)
    else:
        fit_idx, stop_idx = np.arange(len(y)), np.array([], dtype=int)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X[fit_idx]), torch.from_numpy(y[fit_idx]).float()[:, None]),
        batch_size=config.batch_size,
        shuffle=True,
        generator=gen,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    X_stop = torch.from_numpy(X[stop_idx])
    y_stop = torch.from_numpy(y[stop_idx]).float()[:, None]
    best_loss, best_state, bad_epochs, epochs_run = float("inf"), None, 0, 0
    history: list[float] = []
    for _ in range(config.epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        history.append(total / len(fit_idx))
        epochs_run += 1
        if not early_stop:
            continue
        model.eval()
        with torch.no_grad():
            stop_loss = float(loss_fn(model(X_stop), y_stop).item())
        if stop_loss < best_loss:
            best_loss, bad_epochs = stop_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {"epochs_run": epochs_run, "train_loss": history, "early_stopping": early_stop}
    if early_stop:
        info["best_stop_loss"] = best_loss
    return model, info


def _frame_counts(pred: np.ndarray, truth: np.ndarray) -> Counts:
    return Counts(
        tp=int(((pred == 1) & (truth == 1)).sum()),
        fp=int(((pred == 1) & (truth == 0)).sum()),
        fn=int(((pred == 0) & (truth == 1)).sum()),
    )


def _event_counts(df: pd.DataFrame, pred: np.ndarray, config: TrainConfig) -> Counts | None:
    if config.track_column not in df.columns or "frame" not in df.columns:
        return None
    tmp = df[[config.group_column, config.track_column, "frame", config.label_column]].copy()
    tmp["_pred"] = pred
    total = Counts(0, 0, 0)
    for _, g in tmp.sort_values("frame").groupby([config.group_column, config.track_column]):
        total = total + match_events(
            events_from_frames(g["_pred"].to_numpy()),
            events_from_frames(g[config.label_column].to_numpy()),
        )
    return total


def _counts_dict(c: Counts) -> dict:
    return {"tp": c.tp, "fp": c.fp, "fn": c.fn, "precision": c.precision, "recall": c.recall,
            "f1": c.f1}


def train(
    df: pd.DataFrame, config: TrainConfig, layout: CameraLayout | None = None
) -> TrainResult:
    """Train / evaluate one model on a pixel frame table with a video-grouped split."""
    for col in (config.label_column, config.group_column):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in table")
    y = validate_binary_labels(df[config.label_column], config.label_column)
    groups = df[config.group_column].astype(str).to_numpy()
    train_idx, val_idx = grouped_split(groups, config.test_size, seed=config.seed)
    val_df = df.iloc[val_idx]
    threshold = config.decision_threshold

    info: dict = {}
    if config.model == "rule":
        model: object = None
        pred = predict_table(val_df)
    else:
        roi_left = None
        if config.variant == "generalise":
            if layout is None:
                raise ValueError("variant 'generalise' needs a camera layout (--camera)")
            roi_left = roi_left_from_layout(df, layout, config.track_column)
        X = build_features(df, config.variant, config.engineered, roi_left)
        if config.model == "rf":
            _seed_everything(config.seed)
            model = make_random_forest(config.n_estimators, config.seed, **config.rf_params)
            model.fit(X[train_idx], y[train_idx])
            if 1 not in model.classes_:
                raise ValueError("training split has no positive samples")
            proba = model.predict_proba(X[val_idx])[:, list(model.classes_).index(1)]
        else:
            model, info = _fit_torch(config, X[train_idx], y[train_idx], groups[train_idx])
            proba = _predict_proba_torch(model, X[val_idx])
        pred = (proba >= threshold).astype(int)

    y_val = y[val_idx]
    metrics = {
        "model": config.model,
        "threshold": threshold,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "val_groups": sorted(set(groups[val_idx].tolist())),
        "positive_rate_val": float(y_val.mean()) if len(y_val) else 0.0,
        "frame": _counts_dict(_frame_counts(pred, y_val)),
        **info,
    }
    events = _event_counts(val_df, pred, config)
    if events is not None:
        metrics["event"] = _counts_dict(events)
    return TrainResult(config=config, model=model, metrics=metrics)


def save_model(result: TrainResult, path: str | Path) -> Path:
    """Persist a trained model with its config (torch ``state_dict`` or joblib for the RF)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = asdict(result.config)
    if isinstance(result.model, nn.Module):
        torch.save({"config": cfg, "state_dict": result.model.state_dict()}, path)
    elif result.model is not None:
        import joblib

        joblib.dump({"config": cfg, "model": result.model}, path)
    else:
        raise ValueError("the rule baseline has no model to save")
    return path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="train on generated data")
    src.add_argument("--data", nargs="+", help="per-video CSV files (pixel kp_{k}_x/y or (x,y))")
    p.add_argument("--config", help="YAML file with TrainConfig keys")
    p.add_argument("--camera", help="camera layout YAML (needed for --variant generalise)")
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--variant", choices=VARIANTS)
    p.add_argument("--engineered", action="store_true", default=None)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--patience", type=int)
    p.add_argument("--test-size", dest="test_size", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--threshold", type=float)
    p.add_argument("--label-column", dest="label_column")
    p.add_argument("--n-estimators", dest="n_estimators", type=int)
    p.add_argument("--n-videos", type=int, default=8, help="synthetic: number of videos")
    p.add_argument("--n-desks", type=int, default=4, help="synthetic: desks per video")
    p.add_argument("--n-frames", type=int, default=150, help="synthetic: frames per desk")
    p.add_argument("--save", help="write the trained model here")
    return p


def resolve_config(args: argparse.Namespace) -> TrainConfig:
    """Defaults < YAML (``--config``) < explicit CLI flags."""
    values: dict = {}
    if args.config:
        with open(args.config, encoding="utf8") as f:
            values.update(yaml.safe_load(f) or {})
    for f in fields(TrainConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            values[f.name] = v
    return TrainConfig.from_dict(values)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = resolve_config(args)
    if args.synthetic:
        df = generate_frame_table(
            "hand_raise", n_videos=args.n_videos, n_desks=args.n_desks,
            n_frames=args.n_frames, seed=config.seed,
        )
        if config.label_column not in df.columns:
            df = df.rename(columns={"hand_raise": config.label_column})
    else:
        df = load_tables(args.data, config.label_column, config.group_column)
    layout = CameraLayout.from_yaml(args.camera) if args.camera else None
    result = train(df, config, layout)
    if args.save:
        result.metrics["saved_to"] = str(save_model(result, args.save))
    json.dump(result.metrics, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
