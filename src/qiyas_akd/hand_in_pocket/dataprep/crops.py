"""Grayscale desk crops for the image and hybrid models, behind one small interface.

Image paths are deployment data, so the models never see a path: they get a :class:`CropProvider`
that returns one grayscale crop in [0, 1] for ``(video, desk_no, frame)``.

* :class:`FolderCropProvider` — crops exported to disk (one file per desk and frame; the file-name
  template and the sub-folders to search are configurable, e.g. per-class folders).
* :class:`RenderedCropProvider` — synthetic crops for running and testing without data: the desk's
  keypoints are drawn as bright dots on a noisy background, inside a per-track bounding box, so the
  image models have a learnable (if trivial) signal.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .preprocess import frame_xy


class CropProvider(Protocol):
    size: int

    def get(self, video: str, desk_no: int, frame: int) -> np.ndarray:
        """(size, size) float32 grayscale crop in [0, 1]."""
        ...


class FolderCropProvider:
    """Load ``root/<subdir>/<pattern>`` (first sub-folder holding it), resize, scale to [0, 1].

    Needs Pillow (installed with matplotlib); imported lazily.
    """

    def __init__(
        self,
        root: str | Path,
        size: int = 64,
        pattern: str = "{video}_d{desk_no}_f{frame}.jpg",
        subdirs: Sequence[str] = ("",),
    ) -> None:
        self.root = Path(root)
        self.size = size
        self.pattern = pattern
        self.subdirs = tuple(subdirs)

    def get(self, video: str, desk_no: int, frame: int) -> np.ndarray:
        from PIL import Image

        name = self.pattern.format(video=video, desk_no=int(desk_no), frame=int(frame))
        for sub in self.subdirs:
            p = self.root / sub / name
            if p.is_file():
                img = Image.open(p).convert("L").resize((self.size, self.size))
                return np.asarray(img, dtype=np.float32) / 255.0
        raise FileNotFoundError(f"crop {name} not found under {self.root} in {self.subdirs}")


class RenderedCropProvider:
    """Synthetic crops rendered from a per-frame keypoint table (pixel coordinates)."""

    def __init__(
        self, frames: pd.DataFrame, size: int = 64, noise: float = 0.05, margin: float = 20
    ):
        self.size = size
        self.noise = noise
        xy = frame_xy(frames)
        valid = ~(xy == 0).all(axis=-1)
        self._xy: dict[tuple[str, int, int], np.ndarray] = {}
        boxes: dict[tuple[str, int], list[float]] = {}
        keys = zip(frames["video"].astype(str), frames["desk_no"], frames["frame"], strict=True)
        for i, (v, d, f) in enumerate(keys):
            pts = xy[i][valid[i]]
            self._xy[(v, int(d), int(f))] = pts
            if len(pts):
                b = boxes.setdefault((v, int(d)), [np.inf, np.inf, -np.inf, -np.inf])
                b[0], b[1] = min(b[0], pts[:, 0].min()), min(b[1], pts[:, 1].min())
                b[2], b[3] = max(b[2], pts[:, 0].max()), max(b[3], pts[:, 1].max())
        self._box = {
            k: (b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin)
            for k, b in boxes.items()
        }
        self._cache: dict[tuple[str, int, int], np.ndarray] = {}

    def get(self, video: str, desk_no: int, frame: int) -> np.ndarray:
        key = (str(video), int(desk_no), int(frame))
        if key in self._cache:
            return self._cache[key]
        seed = zlib.crc32(f"{key[0]}|{key[1]}|{key[2]}".encode())
        rng = np.random.default_rng(seed)
        img = rng.uniform(0, self.noise, (self.size, self.size)).astype(np.float32)
        pts = self._xy.get(key)
        box = self._box.get(key[:2])
        if pts is not None and len(pts) and box is not None:
            x0, y0, x1, y1 = box
            u = ((pts[:, 0] - x0) / max(x1 - x0, 1e-6) * (self.size - 1)).astype(int)
            w = ((pts[:, 1] - y0) / max(y1 - y0, 1e-6) * (self.size - 1)).astype(int)
            img[np.clip(w, 0, self.size - 1), np.clip(u, 0, self.size - 1)] = 1.0
        self._cache[key] = img
        return img
