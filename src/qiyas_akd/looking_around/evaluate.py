"""CLI: event-level evaluation of a looking-around configuration on a labelled frame table.

Usage::

    python -m qiyas_akd.looking_around.evaluate --synthetic
    python -m qiyas_akd.looking_around.evaluate --synthetic --config ear_line_linear.yaml
    python -m qiyas_akd.looking_around.evaluate --data table.csv --camera camera.yaml

``--data`` expects the frame-table layout of ``synthetic.generate_frame_table`` (``video, frame,
desk_no, kp_{k}_x, kp_{k}_y`` in pixels, plus the 0/1 label column). Prints TP / FP / FN and P / R /
F1 as JSON. With ``--synthetic`` and no ``--camera`` the matching ``detector.synthetic_layout`` is
used.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import pandas as pd
import yaml

from ..desks import CameraLayout
from ..synthetic import generate_frame_table
from .detector import LABEL_COLUMN, LookingAroundConfig, evaluate_table, synthetic_layout


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a looking-around rule (event level).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="evaluate on generated data")
    src.add_argument("--data", help="frame-table CSV")
    p.add_argument("--config", help="YAML with LookingAroundConfig keys")
    p.add_argument("--camera", help="camera layout YAML (required with --data)")
    p.add_argument("--label-column", default=LABEL_COLUMN)
    p.add_argument("--tolerance", type=int, default=0, help="event matching tolerance (frames)")
    p.add_argument("--n-videos", type=int, default=4)
    p.add_argument("--n-desks", type=int, default=4)
    p.add_argument("--n-frames", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg_dict: dict = {}
    if args.config:
        with open(args.config, encoding="utf8") as f:
            cfg_dict = yaml.safe_load(f) or {}
    config = LookingAroundConfig.from_dict(cfg_dict)
    if args.synthetic:
        df = generate_frame_table(
            "looking_around", n_videos=args.n_videos, n_desks=args.n_desks,
            n_frames=args.n_frames, seed=args.seed,
        )
        layout = (
            CameraLayout.from_yaml(args.camera) if args.camera else synthetic_layout(args.n_desks)
        )
    else:
        if not args.camera:
            raise SystemExit("--camera is required with --data")
        df = pd.read_csv(args.data)
        df.columns = [str(c).strip() for c in df.columns]
        layout = CameraLayout.from_yaml(args.camera)
    counts = evaluate_table(df, layout, config, args.label_column, args.tolerance)
    json.dump(
        {"method": config.method, "tp": counts.tp, "fp": counts.fp, "fn": counts.fn,
         "precision": counts.precision, "recall": counts.recall, "f1": counts.f1},
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
