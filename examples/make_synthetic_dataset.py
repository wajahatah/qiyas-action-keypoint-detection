"""Write synthetic per-desk keypoint tables for all three behaviours (README quick-start).

The real exam-hall data cannot be published, so the training pipelines are demonstrated on
synthetic sequences with the same long format as the real annotation CSVs (and as
``qiyas_akd.extract``)::

    python examples/make_synthetic_dataset.py              # -> data/synthetic/<behaviour>.csv
    python examples/make_synthetic_dataset.py --out my_synthetic --videos 10 --frames 300

The default output folder ``data/`` is git-ignored. Each behaviour gets its own seed
(``seed + index``) so the three tables are independent but reproducible.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from qiyas_akd.synthetic import BEHAVIOURS, generate_frame_table


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic keypoint CSVs")
    ap.add_argument("--out", default="data/synthetic", help="output folder (default: %(default)s)")
    ap.add_argument("--videos", type=int, default=6, help="videos per behaviour")
    ap.add_argument("--desks", type=int, default=4, help="desks per video")
    ap.add_argument("--frames", type=int, default=120, help="frames per video")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--behaviours", nargs="+", choices=BEHAVIOURS, default=list(BEHAVIOURS), metavar="B"
    )
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, behaviour in enumerate(BEHAVIOURS):
        if behaviour not in a.behaviours:
            continue
        table = generate_frame_table(
            behaviour, n_videos=a.videos, n_desks=a.desks, n_frames=a.frames, seed=a.seed + i
        )
        path = out / f"{behaviour}.csv"
        table.to_csv(path, index=False)
        print(f"{path}: {len(table)} rows, positive rate {table[behaviour].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
