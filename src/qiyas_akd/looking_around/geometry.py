"""Head-orientation geometry from the head and ear keypoints (pixel space, 1280x720).

Three single-frame cues were tried for "looking around" before the ear-line angle won the
offline comparison:

* **ear-line angle** — orientation of the line left_ear → right_ear. With the custom skeleton the
  ``left_ear`` label is the ear on the *image* left, so a candidate facing the camera has a level
  ear line (≈ 0°). Turning or tilting the head rotates that line; once the ears cross over, the
  angle approaches ±180°. The decision always uses ``abs(angle)``.
* **centre angle** — direction of the head point as seen from the ear midpoint, measured from
  "straight up" in the image (the original added +90° to an ``arctan2`` measured from the x axis).
* **visual-field triangle** — drop a perpendicular from the head onto the ear line (foot point D),
  open a cone of half-angle θ upwards from D, and flag looking around when the head leaves the cone.
  Inside/outside is the classic area-sum test.

A distance-based field of view ``θ = 2·arccot(d / k)`` was also explored to widen the cone for
candidates close to the camera and narrow it for distant ones.

Missing points: the pose stage encodes an undetected keypoint as ``(0, 0)``. A missing ear turns
the ear line into a line to the image corner and produces a huge, meaningless angle — this caused
false alerts until the decision was guarded so that **no angle is computed when either ear is
missing**. Every function here returns ``None`` instead of a number in that case, and the detector
treats ``None`` as "no evidence".
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Point = tuple[float, float]

# Length (pixels) of the visual-field cone edges used by the triangle experiments.
DEFAULT_CONE_LENGTH = 150.0
# Scale constant of the distance-based field of view (pixels).
DEFAULT_FOV_K = 100.0


def is_missing(p: Sequence[float] | np.ndarray) -> bool:
    """True for the pose stage's "not detected" encoding: ``(0, 0)`` or any NaN coordinate."""
    x, y = float(p[0]), float(p[1])
    return math.isnan(x) or math.isnan(y) or (x == 0.0 and y == 0.0)


def _wrap_degrees(angle: float) -> float:
    """Map an angle to ``(-180, 180]``."""
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def ear_line_angle(left_ear: Sequence[float], right_ear: Sequence[float]) -> float | None:
    """Angle (degrees, ``(-180, 180]``) of the vector left_ear → right_ear; ``None`` if an ear is
    missing.

    Image y grows downwards, so a positive angle means the right ear is lower than the left one.
    The original helper returned 0 whenever the two x coordinates were exactly equal (a guard for an
    unused slope division); ``arctan2`` is defined there, so the guard is dropped and a vertical ear
    line correctly reports ±90° instead of 0°.
    """
    if is_missing(left_ear) or is_missing(right_ear):
        return None
    dx = float(right_ear[0]) - float(left_ear[0])
    dy = float(right_ear[1]) - float(left_ear[1])
    if dx == 0.0 and dy == 0.0:
        return None
    return math.degrees(math.atan2(dy, dx))


def ear_midpoint(left_ear: Sequence[float], right_ear: Sequence[float]) -> Point | None:
    """Midpoint of the two ears, or ``None`` if either is missing."""
    if is_missing(left_ear) or is_missing(right_ear):
        return None
    return (
        (float(left_ear[0]) + float(right_ear[0])) / 2.0,
        (float(left_ear[1]) + float(right_ear[1])) / 2.0,
    )


def centre_angle(
    head: Sequence[float], left_ear: Sequence[float], right_ear: Sequence[float]
) -> float | None:
    """Deviation (degrees) of the head point from "straight up" as seen from the ear midpoint.

    Same convention as the original: ``arctan2(head - mid) + 90°``, so a head directly above the
    ear midpoint gives 0°, a head displaced to the image right is positive and to the left negative.
    The result is wrapped to ``(-180, 180]``; the original left it in ``(-90°, 270°]``, which only
    differs for heads *below* the ear line and never changes a decision for thresholds ≤ 90°.
    """
    mid = ear_midpoint(left_ear, right_ear)
    if mid is None or is_missing(head):
        return None
    dx = float(head[0]) - mid[0]
    dy = float(head[1]) - mid[1]
    if dx == 0.0 and dy == 0.0:
        return None
    return _wrap_degrees(math.degrees(math.atan2(dy, dx)) + 90.0)


def perpendicular_foot(p: Sequence[float], a: Sequence[float], b: Sequence[float]) -> Point:
    """Foot D of the perpendicular from ``p`` onto the (infinite) line through ``a`` and ``b``.

    ``D = A + t·AB`` with ``t = (AP·AB) / |AB|²``. Raises ``ValueError`` when ``a == b`` (no line);
    the original silently reused ``t`` from the previous person in that case.
    """
    ax, ay = float(a[0]), float(a[1])
    abx, aby = float(b[0]) - ax, float(b[1]) - ay
    denom = abx * abx + aby * aby
    if denom == 0.0:
        raise ValueError("a and b coincide; the line is undefined")
    t = ((float(p[0]) - ax) * abx + (float(p[1]) - ay) * aby) / denom
    return (ax + t * abx, ay + t * aby)


def visual_field(
    apex: Sequence[float], half_angle_deg: float, length: float
) -> tuple[Point, Point]:
    """End points ``(left, right)`` of a cone opening *upwards* (towards smaller y) from ``apex``.

    ``left = apex + (-L·sin θ, -L·cos θ)``, ``right = apex + (L·sin θ, -L·cos θ)``. The original
    used ``π ≈ 3.142`` and rounded the end points to integers for drawing; exact radians and float
    points are used here (sub-pixel difference).
    """
    theta = math.radians(half_angle_deg)
    sx, cy = length * math.sin(theta), length * math.cos(theta)
    x0, y0 = float(apex[0]), float(apex[1])
    return (x0 - sx, y0 - cy), (x0 + sx, y0 - cy)


def triangle_area(p1: Sequence[float], p2: Sequence[float], p3: Sequence[float]) -> float:
    """Unsigned area of a triangle (shoelace formula)."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0


def point_in_triangle(
    p: Sequence[float],
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
    rel_tol: float = 1e-9,
) -> bool:
    """Area-sum test: ``p`` is inside (or on the boundary of) ``abc`` iff the three sub-triangles
    ``pab + pbc + pca`` add up to ``abc``.

    The original compared the sums with ``==``, which only worked because every vertex had been
    rounded to integer pixels; a relative tolerance is required for float coordinates.
    """
    whole = triangle_area(a, b, c)
    parts = triangle_area(p, a, b) + triangle_area(p, b, c) + triangle_area(p, c, a)
    return math.isclose(parts, whole, rel_tol=rel_tol, abs_tol=1e-9)


def head_in_visual_field(
    head: Sequence[float],
    left_ear: Sequence[float],
    right_ear: Sequence[float],
    half_angle_deg: float,
    length: float = DEFAULT_CONE_LENGTH,
) -> bool | None:
    """Triangle method: is the head inside the visual-field cone anchored on the ear line?

    The apex is the foot D of the perpendicular from the head onto the ear line, so for an upright,
    frontal head the head lies straight above D. Returns ``None`` if the head or an ear is missing
    or the ears coincide.
    """
    if is_missing(head) or is_missing(left_ear) or is_missing(right_ear):
        return None
    try:
        d = perpendicular_foot(head, left_ear, right_ear)
    except ValueError:
        return None
    left, right = visual_field(d, half_angle_deg, length)
    return point_in_triangle(head, left, right, d)


def field_of_view(distance: float, k: float = DEFAULT_FOV_K) -> float:
    """Distance-based field of view ``θ = 2·arccot(d / k)`` in degrees.

    ``d`` is the head's distance (pixels) from the top of its desk ROI: small ``d`` (candidate far
    from the camera, near the ROI top) gives a wide angle approaching 180°, large ``d`` narrows it.
    ``arccot(d/k)`` is computed as ``atan2(k, d)`` so ``d = 0`` is well defined (180°).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if distance < 0:
        raise ValueError("distance must be non-negative")
    return math.degrees(2.0 * math.atan2(k, distance))
