"""Config-driven fine-tuning of a YOLO-pose model on the 10-keypoint skeleton (Ultralytics).

The original experiments were a sequence of hand-edited ``model.train(...)`` calls (≈15 runs:
YOLOv8l-pose → YOLO11 l/m/s-pose → YOLO11l-pose). Here every hyper-parameter lives in a YAML file
(``configs/pose/*.yaml``) so a run is reproducible from the config alone, and this module only maps
the config onto Ultralytics' ``train`` arguments.

Two lessons are encoded in the defaults and warnings:

* ``optimizer: auto`` (used in every original run) makes Ultralytics choose the optimizer, ``lr0``
  and ``momentum`` itself and **ignore** the configured ``lr0``/``momentum``. The ``lr0`` sweeps
  between runs (0.01 → 0.0001 → 0.001) therefore had no effect on the optimiser actually used. The
  config keeps ``auto`` to reproduce the runs; :func:`train` logs a warning when that combination
  is configured, so set an explicit optimizer (e.g. ``AdamW``) for a real ``lr0`` study.
* horizontal-flip augmentation needs a correct ``flip_idx`` in ``data.yaml`` — see
  :mod:`qiyas_akd.pose.dataset`.

Ultralytics is imported lazily so this module can be imported, configured and tested without it.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from ._optional import import_yolo

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    """Training hyper-parameters. Field names are Ultralytics ``train`` argument names.

    ``batch`` may be an int (images per batch) or a float in (0, 1) — Ultralytics AutoBatch then
    sizes the batch to use that fraction of GPU memory. ``freeze`` is the number of leading layers
    kept frozen (``None`` = train everything). ``extra`` passes any further Ultralytics argument
    verbatim.
    """

    model: str
    data: str | None = None
    epochs: int = 300
    imgsz: int = 640
    batch: int | float = 16
    patience: int = 25
    optimizer: str = "auto"
    lr0: float = 0.001
    freeze: int | None = None
    seed: int = 0
    deterministic: bool = True
    workers: int = 8
    device: int | str | None = None
    project: str | None = None
    name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.imgsz < 32 or self.patience < 0:
            raise ValueError("epochs >= 1, imgsz >= 32 and patience >= 0 are required")
        if isinstance(self.batch, float) and not (0.0 < self.batch < 1.0):
            raise ValueError("a float batch is a GPU-memory fraction and must be in (0, 1)")
        if isinstance(self.batch, int) and self.batch < 1 and self.batch != -1:
            raise ValueError("an int batch must be >= 1 (or -1 for AutoBatch)")
        clash = set(self.extra) & {f.name for f in fields(self)}
        if clash:
            raise ValueError(f"extra repeats first-class fields: {sorted(clash)}")


def load_train_config(path: str | Path, **overrides: Any) -> TrainConfig:
    """Read a ``configs/pose/*.yaml`` file; ``overrides`` (e.g. ``data=...``) win over the file."""
    with open(path, encoding="utf8") as f:
        raw = yaml.safe_load(f) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)} (put extras under 'extra')")
    return TrainConfig(**raw)


def to_ultralytics_kwargs(cfg: TrainConfig) -> dict[str, Any]:
    """Keyword arguments for ``YOLO(cfg.model).train(**kwargs)`` (``None`` values dropped)."""
    if not cfg.data:
        raise ValueError("TrainConfig.data (path to data.yaml) is required for training")
    kw = asdict(cfg)
    kw.pop("model")
    extra = kw.pop("extra")
    kw = {k: v for k, v in kw.items() if v is not None}
    kw.update(extra)
    return kw


def train(cfg: TrainConfig) -> Any:
    """Fine-tune ``cfg.model`` and return the Ultralytics training results object."""
    kwargs = to_ultralytics_kwargs(cfg)
    if cfg.optimizer.lower() == "auto":
        log.warning(
            "optimizer=auto: Ultralytics will ignore lr0=%s and pick lr0/momentum itself", cfg.lr0
        )
    yolo = import_yolo()
    return yolo(cfg.model).train(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m qiyas_akd.pose.train --config configs/pose/yolo11l_pose.yaml --data D``."""
    ap = argparse.ArgumentParser(description="Fine-tune a YOLO-pose model from a YAML config")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", help="path to data.yaml (overrides the config)")
    ap.add_argument("--device", help="e.g. 0, 0,1 or cpu (overrides the config)")
    ap.add_argument("--name", help="run name (overrides the config)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cfg = load_train_config(a.config, data=a.data, device=a.device, name=a.name)
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
