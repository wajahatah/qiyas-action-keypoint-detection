"""Depth-dependent angle thresholds for the looking-around decision.

Why depth-dependent: one camera sees several desks at different distances and viewing angles, so
the same head turn does not produce the same ear-line rotation everywhere in the frame. The allowed
angle is therefore a function of where the head sits relative to its own desk.

Every threshold here is a callable ``threshold(left_ear, right_ear, desk) -> degrees``; the detector
flags looking around when ``abs(angle) > threshold``.

History of the rule (all values are configurable here):

1. fixed allowed angle (60°; 45° for the first triangle experiments);
2. ``allowed − int(x · 30)`` with a single allowed angle of 60°;
3. per-desk allowed angle chosen by whether the ears are above the desk reference line
   (``base_angle_out``, originally 90°) or below it (``base_angle_in``, originally 75°), later read
   from per-camera calibration;
4. the depth slope ``k`` reduced from 30 to 15 (fewer false alerts for candidates near the camera);
5. a 6th-order polynomial in the depth proxy ``x`` replacing the linear form.

**Depth proxy.** ``x = (ear_mid_y − ref_y) / (roi_ymax − ref_y)`` clipped to ``[0, 1]``: 0 at the
desk reference line, 1 at the bottom of the desk ROI. The first linear versions used
``(roi_ymax − roi_ymin) / 2`` — half the ROI *height*, not a y coordinate — as the reference, which
only coincides with the ROI midline when the ROI starts at y = 0; the reference-line form used here
is the later revision. The originals did not clip ``x``; clipping keeps the linear threshold inside
``[base − k, base]`` and stops the polynomial from extrapolating outside the range it was fitted on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..desks import Desk
from .geometry import DEFAULT_FOV_K, ear_midpoint, field_of_view

# Default polynomial (highest power first) in the depth proxy x, fitted by the author during the
# 2025 threshold experiments (the version kept in the stand-alone ear-angle experiment script).
# It falls from ≈ 50.7° at the reference line (x = 0) to ≈ 21.2° at the ROI bottom (x = 1).
DEFAULT_POLY_COEFFS: tuple[float, ...] = (149.15, -105.85, -212.34, 145.46, 87.967, -93.82, 50.668)


class Threshold(Protocol):
    """Allowed ear-line / centre angle (degrees) for one person at one desk."""

    def __call__(
        self, left_ear: Sequence[float], right_ear: Sequence[float], desk: Desk
    ) -> float: ...


def depth_proxy(ear_mid_y: float, desk: Desk, clip: bool = True) -> float:
    """Normalised depth ``x = (ear_mid_y − desk.ref_y) / (roi_ymax − desk.ref_y)``.

    0 at the desk reference line, 1 at the bottom of the ROI (nearest the camera). Clipped to
    ``[0, 1]`` unless ``clip=False``.
    """
    ymax = float(desk.roi[3])
    ref = float(desk.ref_y)
    if ymax <= ref:
        raise ValueError(f"desk {desk.desk_id}: ROI bottom {ymax} must lie below ref_y {ref}")
    x = (float(ear_mid_y) - ref) / (ymax - ref)
    return float(np.clip(x, 0.0, 1.0)) if clip else x


def ears_outside_band(left_ear: Sequence[float], desk: Desk) -> bool:
    """True when the ears are *above* the desk reference line (the "outside" calibration applies).

    Exact rule of the original: the **left-ear** y is compared with both ends of the reference line
    and the ears count as outside if they are above either end, i.e.
    ``left_ear_y < max(ref_left_y, ref_right_y)``. Without a reference line, ``desk.ref_y`` (the ROI
    top) is used.
    """
    y = float(left_ear[1])
    if desk.ref_line is None:
        return y < float(desk.ref_y)
    return y < max(float(desk.ref_line[1]), float(desk.ref_line[3]))


def _ear_mid_y(left_ear: Sequence[float], right_ear: Sequence[float]) -> float:
    mid = ear_midpoint(left_ear, right_ear)
    if mid is None:
        raise ValueError("threshold needs both ears; check for missing ears before calling it")
    return mid[1]


@dataclass(frozen=True)
class ConstantThreshold:
    """Fixed allowed angle — the first experiments (60° ear line, 45° triangle half-angle)."""

    angle: float = 60.0

    def __call__(self, left_ear: Sequence[float], right_ear: Sequence[float], desk: Desk) -> float:
        return float(self.angle)


@dataclass(frozen=True)
class LinearThreshold:
    """``base − x·k`` with ``x`` the depth proxy.

    * ``base=None`` (default) picks ``desk.base_angle_out`` when the ears are above the reference
      line and ``desk.base_angle_in`` otherwise (see ``ears_outside_band``); a number forces one
      base for every desk (the earliest single-angle version used 60).
    * ``k`` is the depth slope: 30 in the first versions, 15 in the later ones.
    * ``integer_step=True`` reproduces the original ``base − int(x·k)``: the reduction moves in
      whole degrees (truncation; ``x·k ≥ 0`` after clipping, so this is a floor).
    """

    k: float = 15.0
    base: float | None = None
    integer_step: bool = True

    def base_angle(self, left_ear: Sequence[float], desk: Desk) -> float:
        if self.base is not None:
            return float(self.base)
        outside = ears_outside_band(left_ear, desk)
        return float(desk.base_angle_out if outside else desk.base_angle_in)

    def __call__(self, left_ear: Sequence[float], right_ear: Sequence[float], desk: Desk) -> float:
        x = depth_proxy(_ear_mid_y(left_ear, right_ear), desk)
        step = x * self.k
        if self.integer_step:
            step = float(int(step))
        return self.base_angle(left_ear, desk) - step


@dataclass(frozen=True)
class PolynomialThreshold:
    """``polyval(coeffs, x)`` with ``x`` the clipped depth proxy; coefficients highest power first.

    The polynomial is a function of depth alone and replaces the per-desk base angles, so
    ``desk.base_angle_*`` are not used. Supply your own ``coeffs`` after re-fitting on your camera
    geometry; the default is ``DEFAULT_POLY_COEFFS``.
    """

    coeffs: tuple[float, ...] = DEFAULT_POLY_COEFFS
    clip: bool = True

    def __post_init__(self) -> None:
        if len(self.coeffs) == 0:
            raise ValueError("coeffs must not be empty")

    def __call__(self, left_ear: Sequence[float], right_ear: Sequence[float], desk: Desk) -> float:
        x = depth_proxy(_ear_mid_y(left_ear, right_ear), desk, clip=self.clip)
        return float(np.polyval(np.asarray(self.coeffs, dtype=float), x))


@dataclass(frozen=True)
class FieldOfViewThreshold:
    """Distance-based field of view ``2·arccot(d / k)`` (see ``geometry.field_of_view``).

    ``d`` is the distance in pixels from the ear midpoint to the top of the desk ROI. This was an
    exploratory alternative to the linear rule and never went beyond experiments.
    """

    k: float = DEFAULT_FOV_K

    def __call__(self, left_ear: Sequence[float], right_ear: Sequence[float], desk: Desk) -> float:
        d = abs(_ear_mid_y(left_ear, right_ear) - float(desk.roi[1]))
        return field_of_view(d, self.k)


def make_threshold(spec: dict | None) -> Threshold:
    """Build a threshold from a plain config dict, e.g. ``{"kind": "linear", "k": 15}``.

    Kinds: ``constant`` (angle), ``linear`` (k, base, integer_step), ``polynomial`` (coeffs, clip),
    ``fov`` (k). ``None`` gives the default ``LinearThreshold()``.
    """
    if spec is None:
        return LinearThreshold()
    spec = dict(spec)
    kind = spec.pop("kind", "linear")
    if kind == "constant":
        return ConstantThreshold(**spec)
    if kind == "linear":
        return LinearThreshold(**spec)
    if kind == "polynomial":
        if "coeffs" in spec:
            spec["coeffs"] = tuple(float(c) for c in spec["coeffs"])
        return PolynomialThreshold(**spec)
    if kind == "fov":
        return FieldOfViewThreshold(**spec)
    raise ValueError(f"unknown threshold kind {kind!r}")


__all__ = [
    "DEFAULT_POLY_COEFFS",
    "ConstantThreshold",
    "FieldOfViewThreshold",
    "LinearThreshold",
    "PolynomialThreshold",
    "Threshold",
    "depth_proxy",
    "ears_outside_band",
    "make_threshold",
]
