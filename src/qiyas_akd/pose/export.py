"""Export a trained pose model to ONNX or a TensorRT engine (Ultralytics ``export``).

Deployment ran the pose model as a TensorRT FP16 engine, which roughly doubled throughput over the
PyTorch checkpoint. Ultralytics builds an engine in two steps — PyTorch → ONNX (opset 19 in the
original export log) → TensorRT — so ``format="engine"`` covers both; ``format="onnx"`` stops after
the first.

Batch semantics matter for multi-desk cameras:

* ``dynamic=False`` builds a **static** engine: every call must feed exactly ``batch`` images (the
  original batch-5 and batch-10 benchmarks fed one frame repeated 5 or 10 times for that reason);
* ``dynamic=True`` builds an engine with an optimisation profile up to ``batch``, so any batch size
  ``<= batch`` is accepted, at a small cost in peak throughput.

Ultralytics always writes ``<weights-stem>.engine`` next to the weights, so successive exports with
different settings overwrite each other (the originals were renamed by hand afterwards).
:func:`export` renames the artefact to an unambiguous :func:`artefact_name` instead.

Ultralytics is imported lazily so this module can be imported and tested without it.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from ._optional import import_yolo

FORMATS = ("engine", "onnx")


@dataclass(frozen=True)
class ExportConfig:
    """Export settings. Field names (except ``weights``/``rename``) are Ultralytics arguments."""

    weights: str
    format: str = "engine"
    half: bool = True
    batch: int = 10
    dynamic: bool = False
    imgsz: int = 640
    opset: int | None = 19
    device: int | str | None = 0
    workspace: float | None = None
    rename: bool = True

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("weights (trained .pt checkpoint) is required")
        if self.format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}, got {self.format!r}")
        if self.batch < 1:
            raise ValueError("batch must be >= 1")
        if self.imgsz < 32:
            raise ValueError("imgsz must be >= 32")


def load_export_config(path: str | Path, **overrides: Any) -> ExportConfig:
    """Read an export YAML; non-``None`` ``overrides`` win over the file."""
    with open(path, encoding="utf8") as f:
        raw = yaml.safe_load(f) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    unknown = set(raw) - {f.name for f in fields(ExportConfig)}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    return ExportConfig(**raw)


def export_kwargs(cfg: ExportConfig) -> dict[str, Any]:
    """Keyword arguments for ``YOLO(cfg.weights).export(**kwargs)`` (``None`` values dropped)."""
    kw = asdict(cfg)
    kw.pop("weights")
    kw.pop("rename")
    return {k: v for k, v in kw.items() if v is not None}


def artefact_name(cfg: ExportConfig) -> str:
    """Unambiguous file name, e.g. ``best_fp16_b10.engine`` or ``best_fp32_b5_dyn.onnx``."""
    precision = "fp16" if cfg.half else "fp32"
    dyn = "_dyn" if cfg.dynamic else ""
    return f"{Path(cfg.weights).stem}_{precision}_b{cfg.batch}{dyn}.{cfg.format}"


def export(cfg: ExportConfig) -> Path:
    """Export the model; returns the path of the written ONNX file or TensorRT engine."""
    yolo = import_yolo()
    out = Path(yolo(cfg.weights, task="pose").export(**export_kwargs(cfg)))
    if cfg.rename:
        target = out.with_name(artefact_name(cfg))
        out = out.replace(target)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m qiyas_akd.pose.export --config configs/pose/export_tensorrt_fp16.yaml``."""
    ap = argparse.ArgumentParser(description="Export a pose model to ONNX / TensorRT")
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", help="trained .pt checkpoint (overrides the config)")
    ap.add_argument("--batch", type=int, help="overrides the config")
    a = ap.parse_args(argv)
    print(export(load_export_config(a.config, weights=a.weights, batch=a.batch)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
