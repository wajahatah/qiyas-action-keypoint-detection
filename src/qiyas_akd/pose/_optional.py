"""Lazy imports of the optional ``pose`` extra (Ultralytics, OpenCV).

Everything in this repository must import, test and run on synthetic data without a GPU and without
the AGPL Ultralytics stack installed. Heavy dependencies are therefore imported only inside the
functions that really need them, through these helpers, which turn a bare ``ModuleNotFoundError``
into an actionable message.
"""

from __future__ import annotations

from typing import Any

_EXTRA_HINT = "install the optional extra: pip install 'qiyas-akd[pose]'"


def import_yolo() -> Any:
    """Return the ``ultralytics.YOLO`` class or raise an ``ImportError`` that says how to fix it."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            f"Ultralytics is required for pose training/export/inference — {_EXTRA_HINT}"
        ) from exc
    return YOLO


def import_cv2() -> Any:
    """Return the ``cv2`` module or raise an ``ImportError`` that says how to fix it."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(f"OpenCV is required for video reading/resizing — {_EXTRA_HINT}") from exc
    return cv2
