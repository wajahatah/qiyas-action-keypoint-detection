"""Desk / camera layout: which person belongs to which desk, and where the desk sits in the view.

Each camera covers several desks. For every desk the layout stores:

* ``roi`` — desk bounding box ``(xmin, ymin, xmax, ymax)`` in 1280x720 pixels; a detected person is
  assigned to the desk whose ROI contains their reference keypoint;
* ``position`` — a 4-value descriptor of where the desk is in the camera view (e.g.
  left/centre/right, near/far). Perspective changes how "hand in pocket" and "looking around"
  look, so this descriptor is
  appended to the keypoint features;
* ``ref_line`` — a horizontal reference line ``(left_x, left_y, right_x, right_y)`` at the desk
  edge,
  used to compute how far a head is from the camera (a depth proxy for angle thresholds);
* ``base_angle_in`` / ``base_angle_out`` — allowed ear-line angle when the head is inside / outside
  the desk band (per-camera calibration for looking around).

Layouts are plain YAML so no real camera details are needed in code (see ``configs/cameras/``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .keypoints import KP


@dataclass(frozen=True)
class Desk:
    desk_id: int
    roi: tuple[float, float, float, float]
    position: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    ref_line: tuple[float, float, float, float] | None = None
    base_angle_in: float = 75.0
    base_angle_out: float = 90.0

    def __post_init__(self) -> None:
        xmin, ymin, xmax, ymax = self.roi
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"desk {self.desk_id}: invalid roi {self.roi}")
        if len(self.position) != 4:
            raise ValueError(f"desk {self.desk_id}: position must have 4 values")

    def contains(self, x: float, y: float) -> bool:
        xmin, ymin, xmax, ymax = self.roi
        return xmin <= x <= xmax and ymin <= y <= ymax

    @property
    def ref_y(self) -> float:
        """Mean y of the reference line (falls back to the ROI top)."""
        if self.ref_line is None:
            return float(self.roi[1])
        return (self.ref_line[1] + self.ref_line[3]) / 2.0


@dataclass(frozen=True)
class CameraLayout:
    camera_id: str
    desks: tuple[Desk, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, d: dict) -> CameraLayout:
        desks = tuple(
            Desk(
                desk_id=int(x["desk_id"]),
                roi=tuple(float(v) for v in x["roi"]),
                position=tuple(float(v) for v in x.get("position", (0, 0, 0, 0))),
                ref_line=tuple(float(v) for v in x["ref_line"]) if x.get("ref_line") else None,
                base_angle_in=float(x.get("base_angle_in", 75.0)),
                base_angle_out=float(x.get("base_angle_out", 90.0)),
            )
            for x in d["desks"]
        )
        ids = [k.desk_id for k in desks]
        if len(ids) != len(set(ids)):
            raise ValueError(f"camera {d.get('camera_id')}: duplicate desk ids")
        return cls(camera_id=str(d["camera_id"]), desks=desks)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CameraLayout:
        with open(path, encoding="utf8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def desk(self, desk_id: int) -> Desk:
        for k in self.desks:
            if k.desk_id == desk_id:
                return k
        raise KeyError(desk_id)

    def assign(self, xy: np.ndarray) -> int | None:
        """Desk id for one person's keypoints (K, 2 pixels), or None.

        The reference point is the head; if the head is missing, the shoulder midpoint. A person
        whose
        reference point lies in no ROI (e.g. an invigilator walking past) is not assigned.
        """
        xy = np.asarray(xy, dtype=float)
        head = xy[KP["head"]]
        if not (head == 0).all():
            ref = head
        else:
            sh = xy[[KP["left_shoulder"], KP["right_shoulder"]]]
            valid = ~(sh == 0).all(axis=1)
            if not valid.any():
                return None
            ref = sh[valid].mean(axis=0)
        for k in self.desks:
            if k.contains(*ref):
                return k.desk_id
        return None
