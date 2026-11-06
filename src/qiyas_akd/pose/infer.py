"""Pose inference → plain numpy arrays in the repository's keypoint conventions.

Every consumer downstream (desk assignment, feature extraction, behaviour models) works on

* ``xy``   — ``(persons, K, 2)`` pixel coordinates in the 1280x720 frame, with every keypoint whose
  confidence is below :data:`~qiyas_akd.keypoints.CONF_THRESHOLD` (0.5) set to ``(0, 0)``;
* ``conf`` — ``(persons, K)`` raw keypoint confidences (kept, not masked, so a model can still use
  them as features).

This module is the only place that touches Ultralytics result objects, so the rest of the code is
independent of the pose framework (and testable with fake models). The frame is resized to
1280x720 *before* inference — training data and every desk ROI use that resolution, so keypoints
from any camera resolution land in the same coordinate system.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..keypoints import CONF_THRESHOLD, FRAME_H, FRAME_W, NUM_KEYPOINTS, mask_low_confidence
from ._optional import import_cv2, import_yolo


@dataclass(frozen=True)
class PoseDetections:
    """All persons detected in one frame (see module docstring for the conventions)."""

    xy: np.ndarray
    conf: np.ndarray

    def __post_init__(self) -> None:
        if self.xy.ndim != 3 or self.xy.shape[-1] != 2:
            raise ValueError(f"xy must be (persons, K, 2), got {self.xy.shape}")
        if self.conf.shape != self.xy.shape[:2]:
            raise ValueError(f"conf shape {self.conf.shape} does not match xy {self.xy.shape}")

    def __len__(self) -> int:
        return int(self.xy.shape[0])

    @classmethod
    def empty(cls, num_keypoints: int = NUM_KEYPOINTS) -> PoseDetections:
        return cls(np.zeros((0, num_keypoints, 2)), np.zeros((0, num_keypoints)))


def detections_from_array(
    data: Any, threshold: float = CONF_THRESHOLD, num_keypoints: int = NUM_KEYPOINTS
) -> PoseDetections:
    """Split raw ``(persons, K, 3)`` ``[x, y, conf]`` keypoints and mask low-confidence points.

    Accepts numpy arrays and torch tensors (moved to CPU). An empty result (no person, or the
    ``(1, 0, 3)`` shape some exported backends return) gives an empty :class:`PoseDetections`. A
    keypoint count other than ``num_keypoints`` raises: it means a stock COCO-17 model was loaded
    instead of the custom 10-keypoint one, and silently indexing it would scramble every feature.
    """
    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return PoseDetections.empty(num_keypoints)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected (persons, K, 3) keypoints, got {arr.shape}")
    if arr.shape[1] != num_keypoints:
        raise ValueError(
            f"model returned {arr.shape[1]} keypoints, expected {num_keypoints} "
            "(is this the custom 10-keypoint model?)"
        )
    conf = arr[..., 2]
    return PoseDetections(mask_low_confidence(arr[..., :2], conf, threshold), conf)


def detections_from_result(result: Any, threshold: float = CONF_THRESHOLD) -> PoseDetections:
    """Convert one Ultralytics ``Results`` object (``result.keypoints.data``)."""
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return PoseDetections.empty()
    return detections_from_array(keypoints.data, threshold)


def resize_to_frame(frame: np.ndarray, width: int = FRAME_W, height: int = FRAME_H) -> np.ndarray:
    """Resize a BGR frame to 1280x720 (a no-op, needing no OpenCV, if already that size)."""
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return import_cv2().resize(frame, (width, height))


class UltralyticsPoseModel:
    """Callable ``frame -> PoseDetections`` around a YOLO-pose ``.pt`` checkpoint or ``.engine``.

    Ultralytics is imported when the object is created, not when this module is imported.
    """

    def __init__(
        self,
        weights: str,
        threshold: float = CONF_THRESHOLD,
        device: int | str | None = None,
        resize: bool = True,
    ) -> None:
        self.model = import_yolo()(weights, task="pose")
        self.threshold = threshold
        self.device = device
        self.resize = resize

    def predict(self, frames: Sequence[np.ndarray]) -> list[PoseDetections]:
        """Batched inference; returns one :class:`PoseDetections` per frame, in order."""
        batch = [resize_to_frame(f) if self.resize else f for f in frames]
        kwargs: dict[str, Any] = {"verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model(batch, **kwargs)
        return [detections_from_result(r, self.threshold) for r in results]

    def __call__(self, frame: np.ndarray) -> PoseDetections:
        return self.predict([frame])[0]
