"""Pose-dataset preparation for the custom 10-keypoint skeleton (Ultralytics YOLO-pose format).

Three steps, each a fix or a formalisation of what the original dataset scripts did:

1. **Keypoint re-ordering** (``reorder_label_file`` / ``reorder_label_dir``). The annotation tool
   exported the 10 points in its own order (``ANNOTATION_ORDER``); training needs the schema order
   of :data:`qiyas_akd.keypoints.KEYPOINT_NAMES`, which every downstream feature indexes into.
2. **Video-grouped train/val split** (``plan_video_split`` / ``split_by_video``). The original split
   shuffled *individual frames* 80/20. Frames sampled from the same exam video are near-duplicates
   (same candidate, same desk, same lighting, a fraction of a second apart), so almost every
   validation frame had a near-copy in training. Validation mAP therefore measured memorisation, not
   generalisation to an unseen camera/session — the likely reason it looked near-perfect in the
   later model iterations. Grouping by source video keeps each video on exactly one side.
3. **``data.yaml``** (``build_data_yaml`` / ``write_data_yaml``). The original file had
   ``flip_idx: []``. For pose datasets Ultralytics only applies horizontal-flip augmentation when it
   knows which keypoints swap under a mirror; with an empty ``flip_idx`` it sets ``fliplr=0`` at
   dataloader build time and only logs a warning — the run's ``args.yaml`` still records
   ``fliplr: 0.5``, so the loss of augmentation is easy to miss. The default here is the correct
   left/right permutation (``FLIP_IDX``); ``flip="original"`` reproduces the historical behaviour.

Label line format (one detected person per line, all values normalised to [0, 1] except the class
and visibility): ``class cx cy w h  x0 y0 v0  x1 y1 v1 ... x9 y9 v9``.
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from ..evaluation import grouped_split
from ..keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS

# Order in which the annotation tool exported the keypoints (from the original re-ordering script).
ANNOTATION_ORDER: tuple[str, ...] = (
    "head",
    "left_ear",
    "right_ear",
    "neck",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "right_elbow",
    "right_hand",
    "left_hand",
)

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")

# ``<video>_f0001``, ``<video>_frame_0001``, ``<video>-0001``, ``<video>_0001`` → ``<video>``. The
# frame extractor named frames ``<video>_f<index:04d>``; the other spellings cover common tools.
DEFAULT_FRAME_PATTERN = r"^(?P<video>.+?)[_-](?:frame_?|f)?\d+$"

FlipMode = Literal["correct", "original"]


def compute_flip_idx(names: Sequence[str] = KEYPOINT_NAMES) -> list[int]:
    """Index permutation that maps each keypoint to its mirror image (``left_*`` ↔ ``right_*``).

    Ultralytics applies it after a horizontal flip so that, e.g., the point that *was* the left
    hand is relabelled as the right hand. Midline points (head, neck) map to themselves. The result
    is an involution (applying it twice is the identity).
    """
    index = {n: i for i, n in enumerate(names)}
    out = []
    for n in names:
        if n.startswith("left_"):
            mirror = "right_" + n[len("left_") :]
        elif n.startswith("right_"):
            mirror = "left_" + n[len("right_") :]
        else:
            mirror = n
        if mirror not in index:
            raise ValueError(f"keypoint {n!r} has no mirror {mirror!r} in the schema")
        out.append(index[mirror])
    return out


FLIP_IDX: tuple[int, ...] = tuple(compute_flip_idx())


def build_data_yaml(
    root: str | Path,
    train: str = "train/images",
    val: str = "val/images",
    test: str | None = None,
    flip: FlipMode = "correct",
) -> dict:
    """Ultralytics pose dataset description for the 10-keypoint, single-class (person) skeleton.

    ``root`` should be absolute: Ultralytics resolves a relative ``path`` against its own global
    datasets directory, not against the YAML file. ``train``/``val``/``test`` are relative to it and
    point at image folders; labels are found by replacing ``images`` with ``labels`` in the path,
    which is exactly the layout :func:`split_by_video` writes.
    """
    if flip == "correct":
        flip_idx = list(FLIP_IDX)
    elif flip == "original":
        flip_idx = []
    else:
        raise ValueError(f"flip must be 'correct' or 'original', got {flip!r}")
    return {
        "path": str(root),
        "train": train,
        "val": val,
        "test": test,
        "kpt_shape": [NUM_KEYPOINTS, 3],
        "flip_idx": flip_idx,
        "names": {0: "person"},
    }


def write_data_yaml(out_path: str | Path, root: str | Path, **kwargs) -> Path:
    """Write :func:`build_data_yaml` to ``out_path`` with a header listing the keypoint order."""
    data = build_data_yaml(root, **kwargs)
    header = "# 10-keypoint upper-body skeleton, index order:\n" + "".join(
        f"#   {i}: {n}\n" for i, n in enumerate(KEYPOINT_NAMES)
    )
    if not data["flip_idx"]:
        header += "# flip_idx is empty: Ultralytics will disable horizontal-flip augmentation.\n"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf8")
    return out_path


# --------------------------------------------------------------------------------------------------
# Keypoint re-ordering
# --------------------------------------------------------------------------------------------------


def reorder_permutation(
    src_order: Sequence[str], dst_order: Sequence[str] = KEYPOINT_NAMES
) -> list[int]:
    """``perm`` such that ``dst_keypoints[i] = src_keypoints[perm[i]]``."""
    if sorted(src_order) != sorted(dst_order) or len(set(src_order)) != len(src_order):
        raise ValueError("src_order and dst_order must be permutations of the same keypoint names")
    pos = {n: i for i, n in enumerate(src_order)}
    return [pos[n] for n in dst_order]


def reorder_label_line(line: str, src_order: Sequence[str] = ANNOTATION_ORDER) -> str:
    """Re-order the keypoint triplets of one YOLO-pose label line into schema order.

    Output formatting matches the original script: box and coordinates with 6 decimals, the class
    and visibility flag as integers. Blank lines return ``""``.
    """
    values = line.split()
    if not values:
        return ""
    k = len(src_order)
    if len(values) != 5 + 3 * k:
        raise ValueError(f"expected {5 + 3 * k} values (box + {k}x3 keypoints), got {len(values)}")
    cls = int(float(values[0]))
    box = [float(v) for v in values[1:5]]
    kpts = [
        (float(values[i]), float(values[i + 1]), int(float(values[i + 2])))
        for i in range(5, len(values), 3)
    ]
    perm = reorder_permutation(src_order)
    head = f"{cls} " + " ".join(f"{v:.6f}" for v in box)
    body = " ".join(f"{kpts[j][0]:.6f} {kpts[j][1]:.6f} {kpts[j][2]}" for j in perm)
    return f"{head} {body}"


def reorder_label_file(
    src: str | Path, dst: str | Path, src_order: Sequence[str] = ANNOTATION_ORDER
) -> int:
    """Re-order every line of one label file; returns the number of person annotations written."""
    lines = Path(src).read_text(encoding="utf8").splitlines()
    out = []
    for n, line in enumerate(lines, start=1):
        try:
            new = reorder_label_line(line, src_order)
        except ValueError as exc:
            raise ValueError(f"{src}:{n}: {exc}") from exc
        if new:
            out.append(new)
    Path(dst).write_text("".join(f"{x}\n" for x in out), encoding="utf8")
    return len(out)


def reorder_label_dir(
    src_dir: str | Path, dst_dir: str | Path, src_order: Sequence[str] = ANNOTATION_ORDER
) -> int:
    """Re-order all ``*.txt`` label files of ``src_dir`` into ``dst_dir``; returns files written.

    ``dst_dir`` must differ from ``src_dir``: the operation is not idempotent (running it twice
    permutes the points twice), so in-place rewriting would make an interrupted or repeated run
    silently corrupt the labels.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    if src_dir.resolve() == dst_dir.resolve():
        raise ValueError("dst_dir must differ from src_dir (re-ordering is not idempotent)")
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.txt"))
    for f in files:
        reorder_label_file(f, dst_dir / f.name, src_order)
    return len(files)


# --------------------------------------------------------------------------------------------------
# Video-grouped split
# --------------------------------------------------------------------------------------------------


def video_key(filename: str | Path, pattern: str = DEFAULT_FRAME_PATTERN) -> str:
    """Source-video id encoded in a frame file name (``pattern`` needs a ``video`` named group)."""
    stem = Path(filename).stem
    m = re.match(pattern, stem)
    if m is None:
        raise ValueError(
            f"cannot derive a video id from {stem!r} with pattern {pattern!r}; "
            "pass a pattern with a named group 'video' that matches your frame names"
        )
    return m.group("video")


def plan_video_split(
    filenames: Iterable[str | Path],
    val_fraction: float = 0.2,
    seed: int = 42,
    pattern: str = DEFAULT_FRAME_PATTERN,
) -> dict[str, str]:
    """Map each file name to ``"train"`` or ``"val"`` so that no video is on both sides.

    ``val_fraction`` is a fraction of *videos*, not frames: with few long videos the frame ratio can
    differ noticeably from it — the price of an honest validation set. Deterministic for ``seed``.
    """
    names = sorted(str(Path(f).name) for f in filenames)
    groups = [video_key(n, pattern) for n in names]
    train_idx, val_idx = grouped_split(groups, test_size=val_fraction, seed=seed)
    plan = {names[i]: "train" for i in train_idx}
    plan.update({names[i]: "val" for i in val_idx})
    return plan


@dataclass
class SplitSummary:
    """What :func:`split_by_video` did."""

    train_videos: list[str] = field(default_factory=list)
    val_videos: list[str] = field(default_factory=list)
    n_train_images: int = 0
    n_val_images: int = 0
    missing_labels: list[str] = field(default_factory=list)


def split_by_video(
    images_dir: str | Path,
    labels_dir: str | Path,
    out_dir: str | Path,
    val_fraction: float = 0.2,
    seed: int = 42,
    pattern: str = DEFAULT_FRAME_PATTERN,
    move: bool = False,
) -> SplitSummary:
    """Write ``out_dir/{train,val}/{images,labels}`` with a video-grouped split.

    Files are copied by default (the original moved them, destroying the source pool, so a split
    could not be redone with another seed). An image without a label file is still placed — YOLO
    treats it as a background image — and reported in ``missing_labels``.
    """
    images_dir, labels_dir, out_dir = Path(images_dir), Path(labels_dir), Path(out_dir)
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    plan = plan_video_split(images, val_fraction=val_fraction, seed=seed, pattern=pattern)
    transfer = shutil.move if move else shutil.copy2
    summary = SplitSummary()
    for side in ("train", "val"):
        (out_dir / side / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / side / "labels").mkdir(parents=True, exist_ok=True)
    videos: dict[str, set[str]] = {"train": set(), "val": set()}
    for name, side in sorted(plan.items()):
        transfer(str(images_dir / name), str(out_dir / side / "images" / name))
        label = labels_dir / (Path(name).stem + ".txt")
        if label.exists():
            transfer(str(label), str(out_dir / side / "labels" / label.name))
        else:
            summary.missing_labels.append(name)
        videos[side].add(video_key(name, pattern))
        if side == "train":
            summary.n_train_images += 1
        else:
            summary.n_val_images += 1
    summary.train_videos = sorted(videos["train"])
    summary.val_videos = sorted(videos["val"])
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m qiyas_akd.pose.dataset {data-yaml,reorder,split} ...``."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    dy = sub.add_parser("data-yaml", help="write an Ultralytics data.yaml")
    dy.add_argument("--root", required=True, help="dataset root (absolute path recommended)")
    dy.add_argument("--out", required=True)
    dy.add_argument("--flip", choices=["correct", "original"], default="correct")
    ro = sub.add_parser("reorder", help="re-order keypoints from annotation to schema order")
    ro.add_argument("--src", required=True)
    ro.add_argument("--dst", required=True)
    sp = sub.add_parser("split", help="video-grouped train/val split")
    sp.add_argument("--images", required=True)
    sp.add_argument("--labels", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--val-fraction", type=float, default=0.2)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--pattern", default=DEFAULT_FRAME_PATTERN)
    sp.add_argument("--move", action="store_true", help="move instead of copy")
    a = ap.parse_args(argv)
    if a.cmd == "data-yaml":
        print(write_data_yaml(a.out, a.root, flip=a.flip))
    elif a.cmd == "reorder":
        print(f"re-ordered {reorder_label_dir(a.src, a.dst)} label files")
    else:
        s = split_by_video(
            a.images, a.labels, a.out, a.val_fraction, a.seed, a.pattern, move=a.move
        )
        print(
            f"train: {s.n_train_images} images / {len(s.train_videos)} videos; "
            f"val: {s.n_val_images} images / {len(s.val_videos)} videos; "
            f"missing labels: {len(s.missing_labels)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
