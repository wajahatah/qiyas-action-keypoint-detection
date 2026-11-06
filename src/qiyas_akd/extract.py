"""Video → per-frame, per-desk keypoint table (the input format of every behaviour model).

For each frame the pose model returns every detected person; each person is assigned to a desk with
:meth:`qiyas_akd.desks.CameraLayout.assign` and written as one row in the same long format that
:func:`qiyas_akd.synthetic.generate_frame_table` produces (minus the label column, which comes from
annotation)::

    video, frame, desk_no, position_a..position_d, kp_0_x, kp_0_y, ..., kp_9_x, kp_9_y
    [, kp_0_conf, ..., kp_9_conf]

Coordinates are pixels in the 1280x720 frame, with low-confidence keypoints already ``(0, 0)``.

Decisions (relative to the original extraction scripts):

* **Desk assignment** uses the head point inside the desk's 2-D ROI, falling back to the shoulder
  midpoint when the head is missing. The original compared only the head *x* against each desk's
  x-range; a missing head (masked to ``(0, 0)``) therefore landed in the desk whose x-range
  contains 0, if any, injecting a bogus pose into that desk's sequence. People in no ROI (e.g. an
  invigilator walking past) are dropped, as before.
* **One row per desk and frame** (``duplicates="best"``, default): if two people fall in one ROI
  (a neighbour leaning over, a passer-by) the person with the highest mean keypoint confidence is
  kept, so each desk yields a clean time series. ``duplicates="all"`` reproduces the original,
  which wrote every person.
* Rows are ordered by frame, then desk (the original re-numbered persons by desk for the same
  reason: stable per-desk ordering). The model's arbitrary ``person_idx`` is not written.

The pose model is injected as a callable ``frame -> PoseDetections`` so this module runs without
Ultralytics; OpenCV is imported only when frames are read from a real video file.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .desks import CameraLayout
from .keypoints import NUM_KEYPOINTS
from .pose._optional import import_cv2
from .pose.infer import PoseDetections, resize_to_frame

PoseFn = Callable[[np.ndarray], PoseDetections]
Duplicates = Literal["best", "all"]

POSITION_COLUMNS: tuple[str, ...] = tuple(f"position_{c}" for c in "abcd")
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")


def frame_table_columns(include_conf: bool = False) -> list[str]:
    """Column order of the extracted table (matches the synthetic generator's feature columns)."""
    cols = ["video", "frame", "desk_no", *POSITION_COLUMNS]
    cols += [f"kp_{k}_{c}" for k in range(NUM_KEYPOINTS) for c in ("x", "y")]
    if include_conf:
        cols += [f"kp_{k}_conf" for k in range(NUM_KEYPOINTS)]
    return cols


def iter_video_frames(
    path: str | Path, stride: int = 1, resize: bool = True
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, BGR frame)`` from a video file, resized to 1280x720.

    ``frame_index`` is the index in the source video, so it stays meaningful with ``stride > 1``.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    cv2 = import_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {path}")
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                yield index, resize_to_frame(frame) if resize else frame
            index += 1
    finally:
        cap.release()


def rows_for_frame(
    video: str,
    frame_index: int,
    detections: PoseDetections,
    layout: CameraLayout,
    include_conf: bool = False,
    duplicates: Duplicates = "best",
) -> list[dict]:
    """Desk-assigned rows for one frame's detections (sorted by desk number)."""
    if duplicates not in ("best", "all"):
        raise ValueError(f"duplicates must be 'best' or 'all', got {duplicates!r}")
    if detections.xy.shape[1:] != (NUM_KEYPOINTS, 2):
        raise ValueError(f"expected (persons, {NUM_KEYPOINTS}, 2) keypoints")
    candidates: list[tuple[int, float, int]] = []  # (desk_id, mean conf, person index)
    for p in range(len(detections)):
        desk_id = layout.assign(detections.xy[p])
        if desk_id is not None:
            candidates.append((desk_id, float(detections.conf[p].mean()), p))
    if duplicates == "best":
        best: dict[int, tuple[int, float, int]] = {}
        for c in candidates:
            if c[0] not in best or c[1] > best[c[0]][1]:
                best[c[0]] = c
        candidates = list(best.values())
    candidates.sort(key=lambda c: (c[0], -c[1]))
    rows = []
    for desk_id, _, p in candidates:
        position = layout.desk(desk_id).position
        row: dict = {"video": video, "frame": int(frame_index), "desk_no": desk_id}
        row.update({name: float(v) for name, v in zip(POSITION_COLUMNS, position, strict=True)})
        for k in range(NUM_KEYPOINTS):
            row[f"kp_{k}_x"] = float(detections.xy[p, k, 0])
            row[f"kp_{k}_y"] = float(detections.xy[p, k, 1])
        if include_conf:
            for k in range(NUM_KEYPOINTS):
                row[f"kp_{k}_conf"] = float(detections.conf[p, k])
        rows.append(row)
    return rows


def extract_frame_table(
    frames: Iterable[tuple[int, np.ndarray]],
    pose_fn: PoseFn,
    layout: CameraLayout,
    video: str,
    include_conf: bool = False,
    duplicates: Duplicates = "best",
) -> pd.DataFrame:
    """Run ``pose_fn`` over ``(frame_index, frame)`` pairs and build the long-format table."""
    rows: list[dict] = []
    for index, frame in frames:
        rows.extend(rows_for_frame(video, index, pose_fn(frame), layout, include_conf, duplicates))
    return pd.DataFrame(rows, columns=frame_table_columns(include_conf))


def extract_video(
    path: str | Path,
    pose_fn: PoseFn,
    layout: CameraLayout,
    video: str | None = None,
    stride: int = 1,
    include_conf: bool = False,
    duplicates: Duplicates = "best",
) -> pd.DataFrame:
    """:func:`extract_frame_table` over a video file; ``video`` defaults to the file stem."""
    name = video if video is not None else Path(path).stem
    frames = iter_video_frames(path, stride=stride)
    return extract_frame_table(frames, pose_fn, layout, name, include_conf, duplicates)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: extract one CSV per video. Needs the ``pose`` extra (Ultralytics + OpenCV)."""
    from .pose.infer import UltralyticsPoseModel

    ap = argparse.ArgumentParser(description="Extract per-desk keypoint tables from videos")
    ap.add_argument("--input", required=True, help="a video file or a folder of videos")
    ap.add_argument("--weights", required=True, help="pose .pt checkpoint or TensorRT .engine")
    ap.add_argument("--camera", required=True, help="camera layout YAML (see configs/cameras/)")
    ap.add_argument("--out", required=True, help="output folder for <video>.csv files")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--conf", action="store_true", help="also write kp_{k}_conf columns")
    ap.add_argument("--duplicates", choices=["best", "all"], default="best")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    src = Path(a.input)
    videos = (
        sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
        if src.is_dir()
        else [src]
    )
    layout = CameraLayout.from_yaml(a.camera)
    model = UltralyticsPoseModel(a.weights, device=a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for v in videos:
        table = extract_video(
            v, model, layout, stride=a.stride, include_conf=a.conf, duplicates=a.duplicates
        )
        table.to_csv(out / f"{v.stem}.csv", index=False)
        print(f"{v.name}: {len(table)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
