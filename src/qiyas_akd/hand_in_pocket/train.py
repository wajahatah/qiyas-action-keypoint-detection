"""Config-driven training for every hand-in-pocket model family.

Usage::

    python -m qiyas_akd.hand_in_pocket.train --config configs/hand_in_pocket/mlp_t3.yaml --synthetic
    python -m qiyas_akd.hand_in_pocket.train --config <cfg> --frames <per-frame.csv> --out <dir>

``--synthetic`` builds the per-frame table with ``qiyas_akd.synthetic`` so everything runs without
the (unpublishable) real data. Real data is a long per-frame CSV with columns ``video, frame,
desk_no, kp_{k}_x, kp_{k}_y`` in 1280x720 pixels, optional ``kp_{k}_conf``, optional
``position_a..d`` (else a camera layout YAML is required), and the label column. ``video`` must be
unique per recording across cameras (e.g. camera and video number combined), because it is the
unit of the train/validation split and of every per-desk track.

Torch recipe (the one that produced the deployed model): ``BCEWithLogitsLoss`` (cross-entropy for
2-class heads), AdamW lr 1e-3 / weight decay 1e-4, ``ReduceLROnPlateau`` (x0.1, patience 10) on the
validation loss, gradient clipping at 1.0, early stopping on the validation loss, best checkpoint
kept. Every knob is in the YAML so the older recipes (Adam, no scheduler, ...) are reproducible.

Intentional deviations from the original training scripts:

* the train/validation split is **by video** (``evaluation.grouped_split``) instead of a random row
  split — overlapping windows of the same video on both sides made validation scores near-perfect;
* augmentation is applied to the training split only;
* losses are averaged per sample, not per batch (the last, smaller batch no longer counts double);
* besides window-level metrics, keypoint models are evaluated **event-level** on the validation
  videos through the streaming predictor — the metric that was used to pick models.

Outputs (never inside the package): ``<out>/model.pt`` or ``<out>/model.joblib`` and
``<out>/metrics.json`` (metrics + the resolved config).
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..desks import CameraLayout, Desk
from ..evaluation import Counts, grouped_split
from ..keypoints import FRAME_H, FRAME_W, MISSING
from ..synthetic import generate_frame_table
from .dataprep.augment import jitter
from .dataprep.crops import CropProvider, FolderCropProvider, RenderedCropProvider
from .dataprep.pipeline import DataConfig, build_windows, finalise, prepare_frames
from .infer import KeypointPredictor, evaluate_events, logits_to_prob, predict_video_table
from .models import INPUT_KIND, TORCH_MODELS, TREE_KINDS, build_torch_model, make_tree_model
from .models.sequence import to_sequences
from .models.trees import grid_search

BEHAVIOUR = "hand_in_pocket"


@dataclass(frozen=True)
class ModelConfig:
    """``kind``: rf | xgboost | lightgbm | mlp | lstm | gru | temporal_cnn | temporal_crop_cnn3d |
    hybrid2d | hybrid3d. ``params`` go to the constructor (sizes inferred from data if omitted)."""

    kind: str = "mlp"
    params: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, list[Any]] | None = None
    grid_cv: int = 5
    missing_value: float = MISSING

    def __post_init__(self) -> None:
        if self.kind not in TREE_KINDS and self.kind not in TORCH_MODELS:
            raise ValueError(f"unknown model kind {self.kind!r}")


@dataclass(frozen=True)
class OptimConfig:
    """Training loop + evaluation settings (YAML block ``train:``)."""

    epochs: int = 500
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "plateau"
    scheduler_factor: float = 0.1
    scheduler_patience: int = 10
    grad_clip: float | None = 1.0
    patience: int = 25
    val_size: float = 0.2
    seed: int = 42
    threshold: float = 0.5
    min_event_frames: int = 5

    def __post_init__(self) -> None:
        if self.optimizer not in ("adamw", "adam"):
            raise ValueError("optimizer must be adamw or adam")
        if self.scheduler not in ("plateau", "none"):
            raise ValueError("scheduler must be plateau or none")


@dataclass(frozen=True)
class SyntheticConfig:
    """Size of the generated dataset used with ``--synthetic``."""

    n_videos: int = 8
    n_desks: int = 4
    n_frames: int = 150
    seed: int = 0


@dataclass(frozen=True)
class ImageConfig:
    """Crops for image/hybrid models; ``crop_root`` is only needed for real data."""

    crop_size: int = 64
    crop_root: str | None = None
    pattern: str = "{video}_d{desk_no}_f{frame}.jpg"
    subdirs: tuple[str, ...] = ("",)


@dataclass(frozen=True)
class TrainConfig:
    name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: OptimConfig = field(default_factory=OptimConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    images: ImageConfig = field(default_factory=ImageConfig)
    frames_csv: str | None = None
    camera_layout: str | None = None
    output_dir: str = "outputs/hand_in_pocket"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        d = dict(d)
        img = dict(d.pop("images", {}) or {})
        if "subdirs" in img:
            img["subdirs"] = tuple(img["subdirs"])
        return cls(
            model=ModelConfig(**(d.pop("model", {}) or {})),
            data=DataConfig.from_dict(d.pop("data", {})),
            train=OptimConfig(**(d.pop("train", {}) or {})),
            synthetic=SyntheticConfig(**(d.pop("synthetic", {}) or {})),
            images=ImageConfig(**img),
            **d,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path, encoding="utf8") as f:
            return cls.from_dict(yaml.safe_load(f))


def synthetic_layout(n_desks: int) -> CameraLayout:
    """Camera layout matching ``synthetic.generate_frame_table`` (desk ROIs around each desk)."""
    xs = np.linspace(160, FRAME_W - 160, n_desks)
    half = (xs[1] - xs[0]) / 2 if n_desks > 1 else FRAME_W / 2
    desks = []
    for i, x in enumerate(xs):
        pos = [0.0] * 4
        pos[i % 4] = 1.0
        roi = (max(0.0, x - half), 0.0, min(float(FRAME_W), x + half), float(FRAME_H))
        desks.append(Desk(desk_id=i + 1, roi=roi, position=tuple(pos)))
    return CameraLayout(camera_id="synthetic", desks=tuple(desks))


def load_frames(cfg: TrainConfig, synthetic: bool) -> tuple[pd.DataFrame, CameraLayout | None]:
    """Per-frame table + camera layout (synthetic or from the paths in the config)."""
    if synthetic:
        s = cfg.synthetic
        frames = generate_frame_table(BEHAVIOUR, s.n_videos, s.n_desks, s.n_frames, s.seed)
        frames = frames.rename(columns={BEHAVIOUR: cfg.data.label_col})
        return frames, synthetic_layout(s.n_desks)
    if not cfg.frames_csv:
        raise ValueError("no frames_csv given (config, --frames) — or use --synthetic")
    frames = pd.read_csv(cfg.frames_csv)
    frames["video"] = frames["video"].astype(str)
    layout = CameraLayout.from_yaml(cfg.camera_layout) if cfg.camera_layout else None
    return frames, layout


class WindowTorchDataset(Dataset):
    """Tensors for one split, shaped for the model's ``INPUT_KIND``."""

    def __init__(
        self,
        kind: str,
        windows: pd.DataFrame,
        data: DataConfig,
        missing_value: float = MISSING,
        crops: CropProvider | None = None,
    ) -> None:
        self.input_kind = INPUT_KIND[kind]
        self.y = torch.tensor(windows[data.label_col].to_numpy(np.float32))
        self.X = torch.tensor(windows[data.feature_columns].to_numpy(np.float32))
        names = data.windowing.time_names
        if self.input_kind == "sequence":
            self.X = torch.from_numpy(to_sequences(windows, names, data.static_cols, missing_value))
        self.crops = crops
        if self.input_kind.startswith("image"):
            if crops is None:
                raise ValueError(f"model kind {kind!r} needs a crop provider")
            frame_cols = [f"frame_{t}" for t in names]
            self.keys = list(
                zip(
                    windows["video"].astype(str),
                    windows["desk_no"].astype(int),
                    windows[frame_cols].to_numpy(np.int64).tolist(),
                    strict=True,
                )
            )

    def __len__(self) -> int:
        return len(self.y)

    def _crops(self, i: int) -> torch.Tensor:
        video, desk, frames = self.keys[i]
        assert self.crops is not None
        return torch.from_numpy(np.stack([self.crops.get(video, desk, f) for f in frames]))

    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        if self.input_kind == "image":
            return self._crops(i), self.y[i]
        if self.input_kind == "image+kp":
            return self._crops(i).unsqueeze(1), self.X[i], self.y[i]
        return self.X[i], self.y[i]


def resolve_params(kind: str, params: dict[str, Any], data: DataConfig, crop_size: int) -> dict:
    """Fill input sizes that follow from the data recipe (explicit config values win)."""
    p = dict(params)
    steps = data.windowing.n_steps
    if kind == "mlp":
        p.setdefault("in_dim", len(data.feature_columns))
    elif kind in ("lstm", "gru"):
        p.setdefault("input_size", 20 + len(data.static_cols))
    elif kind in ("temporal_cnn", "temporal_crop_cnn3d"):
        p.setdefault("frames", steps)
        p.setdefault("img_size", crop_size)
    elif kind in ("hybrid2d", "hybrid3d"):
        p.setdefault("frames", steps)
        p.setdefault("kp_dim", len(data.feature_columns))
        p.setdefault("crop_size", crop_size)
    if "clamp" in p and p["clamp"] is not None:
        p["clamp"] = tuple(p["clamp"])
    return p


def _loss(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if out.ndim == 2 and out.shape[1] == 2:
        return nn.functional.cross_entropy(out, y.long())
    return nn.functional.binary_cross_entropy_with_logits(out.reshape(-1), y)


def _evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, np.ndarray]:
    model.eval()
    total, n, probs = 0.0, 0, []
    with torch.no_grad():
        for *inputs, y in loader:
            out = model(*inputs)
            total += _loss(out, y).item() * len(y)
            n += len(y)
            probs.append(logits_to_prob(out).numpy())
    return total / max(n, 1), np.concatenate(probs) if probs else np.zeros(0)


def fit_torch(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    opt_cfg: OptimConfig,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The training loop described in the module docstring; leaves the best weights in ``model``."""
    gen = torch.Generator().manual_seed(opt_cfg.seed)
    train_dl = DataLoader(train_ds, batch_size=opt_cfg.batch_size, shuffle=True, generator=gen)
    val_dl = DataLoader(val_ds, batch_size=opt_cfg.batch_size)
    opt_cls = torch.optim.AdamW if opt_cfg.optimizer == "adamw" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)
    sched = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=opt_cfg.scheduler_factor, patience=opt_cfg.scheduler_patience
        )
        if opt_cfg.scheduler == "plateau"
        else None
    )
    best, best_state, best_epoch, stale = float("inf"), copy.deepcopy(model.state_dict()), -1, 0
    history: list[dict[str, float]] = []
    for epoch in range(opt_cfg.epochs):
        model.train()
        total, n = 0.0, 0
        for *inputs, y in train_dl:
            opt.zero_grad()
            loss = _loss(model(*inputs), y)
            loss.backward()
            if opt_cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt_cfg.grad_clip)
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        val_loss, _ = _evaluate(model, val_dl)
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_loss": val_loss})
        log(f"epoch {epoch + 1}: train {total / max(n, 1):.4f}  val {val_loss:.4f}")
        if val_loss < best:
            best, best_state, best_epoch, stale = (
                val_loss,
                copy.deepcopy(model.state_dict()),
                epoch,
                0,
            )
        else:
            stale += 1
            if stale >= opt_cfg.patience:
                log("early stopping")
                break
        if sched is not None:
            sched.step(val_loss)
    model.load_state_dict(best_state)
    return {"best_val_loss": best, "best_epoch": best_epoch, "epochs_run": len(history)}


def window_metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    """Window-level accuracy / precision / recall / F1 (for monitoring, not model selection)."""
    pred = prob >= threshold
    y = y.astype(bool)
    c = Counts(int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum()))
    acc = float((pred == y).mean()) if len(y) else 0.0
    return {
        "accuracy": acc,
        "precision": c.precision,
        "recall": c.recall,
        "f1": c.f1,
        "tp": c.tp,
        "fp": c.fp,
        "fn": c.fn,
        "n": int(len(y)),
    }


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def run(
    cfg: TrainConfig,
    synthetic: bool = False,
    out_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train one model end to end; returns (and writes) the metrics dict."""
    torch.manual_seed(cfg.train.seed)
    out = Path(out_dir or Path(cfg.output_dir) / cfg.name)
    out.mkdir(parents=True, exist_ok=True)
    data, kind = cfg.data, cfg.model.kind

    frames, layout = load_frames(cfg, synthetic)
    prepared = prepare_frames(frames, data, layout)
    windows = build_windows(prepared, data)
    if windows.empty:
        raise ValueError("no windows produced — tracks shorter than the window?")
    tr_idx, va_idx = grouped_split(windows["video"].astype(str), cfg.train.val_size, cfg.train.seed)
    train_w, val_w = windows.iloc[tr_idx], windows.iloc[va_idx]
    a = data.augment
    if a.enabled:
        train_w = jitter(
            train_w,
            a.max_px,
            a.times_positive,
            a.times_negative,
            data.label_col,
            a.per_value,
            a.seed,
        )
    train_w, val_w = finalise(train_w, data, layout), finalise(val_w, data, layout)
    log(
        f"windows: train {len(train_w)} (pos {int(train_w[data.label_col].sum())}), "
        f"val {len(val_w)} (pos {int(val_w[data.label_col].sum())})"
    )

    metrics: dict[str, Any] = {
        "name": cfg.name,
        "kind": kind,
        "n_features": len(data.feature_columns),
    }
    y_val = val_w[data.label_col].to_numpy()
    predictor: KeypointPredictor | None = None
    if kind in TREE_KINDS:
        model = make_tree_model(kind, cfg.model.params, cfg.train.seed)
        X_tr = train_w[data.feature_columns].to_numpy(np.float32)
        y_tr = train_w[data.label_col].to_numpy()
        if cfg.model.grid:
            groups = train_w["video"].astype(str).to_numpy()
            model = grid_search(model, cfg.model.grid, X_tr, y_tr, groups, cfg.model.grid_cv)
        else:
            model.fit(X_tr, y_tr)
        predictor = KeypointPredictor(model, kind, data)
        prob = predictor.predict_windows(val_w)
        import joblib

        joblib.dump(
            {
                "model": model,
                "kind": kind,
                "data": data.to_dict(),
                "feature_columns": data.feature_columns,
            },
            out / "model.joblib",
        )
    else:
        params = resolve_params(kind, cfg.model.params, data, cfg.images.crop_size)
        model = build_torch_model(kind, **params)
        crops = None
        if INPUT_KIND[kind].startswith("image"):
            crops = (
                RenderedCropProvider(prepared, cfg.images.crop_size)
                if synthetic
                else _folder_crops(cfg.images)
            )
        mv = cfg.model.missing_value
        train_ds = WindowTorchDataset(kind, train_w, data, mv, crops)
        val_ds = WindowTorchDataset(kind, val_w, data, mv, crops)
        metrics.update(fit_torch(model, train_ds, val_ds, cfg.train, log))
        _, prob = _evaluate(model, DataLoader(val_ds, batch_size=cfg.train.batch_size))
        torch.save(
            {
                "kind": kind,
                "params": params,
                "state_dict": model.state_dict(),
                "data": data.to_dict(),
                "missing_value": mv,
                "feature_columns": data.feature_columns,
            },
            out / "model.pt",
        )
        if INPUT_KIND[kind] in ("tabular", "sequence"):
            predictor = KeypointPredictor(model, kind, data, mv)
    metrics["val_windows"] = window_metrics(y_val, prob, cfg.train.threshold)

    if predictor is not None:
        val_videos = set(val_w["video"].astype(str))
        val_frames = prepared[prepared["video"].astype(str).isin(val_videos)]
        per_frame = predict_video_table(val_frames, predictor, layout, cfg.train.threshold)
        c = evaluate_events(per_frame, data.label_col, cfg.train.min_event_frames)
        metrics["val_events"] = {
            "tp": c.tp,
            "fp": c.fp,
            "fn": c.fn,
            "precision": c.precision,
            "recall": c.recall,
            "f1": c.f1,
            "min_event_frames": cfg.train.min_event_frames,
        }
    metrics["config"] = asdict(cfg)
    metrics = _jsonable(metrics)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf8")
    log(f"saved {out}")
    return metrics


def _folder_crops(img: ImageConfig) -> CropProvider:
    if not img.crop_root:
        raise ValueError("image models on real data need images.crop_root")
    return FolderCropProvider(img.crop_root, img.crop_size, img.pattern, img.subdirs)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="YAML training config")
    ap.add_argument("--synthetic", action="store_true", help="train on generated data")
    ap.add_argument("--out", help="output directory (default: <output_dir>/<name>)")
    ap.add_argument("--frames", help="per-frame CSV (overrides frames_csv)")
    ap.add_argument("--layout", help="camera layout YAML (overrides camera_layout)")
    ap.add_argument("--epochs", type=int, help="override train.epochs (e.g. for smoke runs)")
    args = ap.parse_args(argv)
    cfg = TrainConfig.from_yaml(args.config)
    if args.frames:
        cfg = replace(cfg, frames_csv=args.frames)
    if args.layout:
        cfg = replace(cfg, camera_layout=args.layout)
    if args.epochs is not None:
        cfg = replace(cfg, train=replace(cfg.train, epochs=args.epochs))
    metrics = run(cfg, synthetic=args.synthetic, out_dir=args.out)
    summary = {k: metrics[k] for k in ("val_windows", "val_events") if k in metrics}
    print(json.dumps(summary, indent=2))
    return metrics


if __name__ == "__main__":
    main()
