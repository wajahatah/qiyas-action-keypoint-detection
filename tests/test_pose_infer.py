import numpy as np
import pytest

from qiyas_akd.keypoints import NUM_KEYPOINTS
from qiyas_akd.pose.infer import (
    PoseDetections,
    detections_from_array,
    detections_from_result,
    resize_to_frame,
)


def _raw(persons: int = 2) -> np.ndarray:
    data = np.zeros((persons, NUM_KEYPOINTS, 3))
    data[..., 0] = 100 + np.arange(NUM_KEYPOINTS)
    data[..., 1] = 200
    data[..., 2] = 0.9
    return data


def test_low_confidence_points_become_zero_and_conf_is_kept():
    raw = _raw()
    raw[0, 3, 2] = 0.49
    raw[1, 7, 2] = 0.5  # exactly at the threshold is kept
    det = detections_from_array(raw)
    assert det.xy.shape == (2, NUM_KEYPOINTS, 2) and det.conf.shape == (2, NUM_KEYPOINTS)
    assert det.xy[0, 3].tolist() == [0.0, 0.0]
    assert det.xy[1, 7].tolist() == [107.0, 200.0]
    assert det.conf[0, 3] == pytest.approx(0.49)
    assert len(det) == 2
    stricter = detections_from_array(raw, threshold=0.95)
    assert (stricter.xy == 0).all()


def test_empty_and_wrong_shapes():
    assert len(detections_from_array(np.zeros((0, NUM_KEYPOINTS, 3)))) == 0
    assert len(detections_from_array(np.zeros((1, 0, 3)))) == 0
    with pytest.raises(ValueError, match="10-keypoint"):
        detections_from_array(np.zeros((1, 17, 3)))
    with pytest.raises(ValueError):
        detections_from_array(np.zeros((NUM_KEYPOINTS, 3)))
    with pytest.raises(ValueError):
        PoseDetections(np.zeros((1, NUM_KEYPOINTS, 2)), np.zeros((2, NUM_KEYPOINTS)))


def test_from_result_duck_types_ultralytics_and_torch():
    torch = pytest.importorskip("torch")

    class Keypoints:
        data = torch.tensor(_raw(1))

    class Result:
        keypoints = Keypoints()

    class NoPose:
        keypoints = None

    det = detections_from_result(Result())
    assert det.xy.shape == (1, NUM_KEYPOINTS, 2) and det.xy[0, 0].tolist() == [100.0, 200.0]
    assert len(detections_from_result(NoPose())) == 0


def test_resize_is_noop_at_model_resolution():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert resize_to_frame(frame) is frame  # no OpenCV needed
