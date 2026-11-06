import sys
from pathlib import Path

import numpy as np
import pytest

from qiyas_akd.desks import CameraLayout
from qiyas_akd.extract import (
    extract_frame_table,
    frame_table_columns,
    iter_video_frames,
    rows_for_frame,
)
from qiyas_akd.keypoints import KP, NUM_KEYPOINTS
from qiyas_akd.pose.infer import PoseDetections, detections_from_array
from qiyas_akd.synthetic import generate_frame_table

CAMERA = Path(__file__).resolve().parents[1] / "configs" / "cameras" / "example_camera.yaml"


@pytest.fixture
def layout() -> CameraLayout:
    return CameraLayout.from_yaml(CAMERA)


def _person(head_x: float, head_y: float = 300.0, conf: float = 0.9) -> np.ndarray:
    """(K, 3) raw keypoints: a person whose points sit around the head position."""
    p = np.zeros((NUM_KEYPOINTS, 3))
    p[:, 0] = head_x + np.arange(NUM_KEYPOINTS)
    p[:, 1] = head_y + 10 * np.arange(NUM_KEYPOINTS)
    p[:, 2] = conf
    return p


def _fake_pose(frame: np.ndarray) -> PoseDetections:
    """Scene encoded in the frame's first pixel: frame 0 has people at desks 3 and 1 plus an
    out-of-ROI walker; frame 1 has a person whose head is missing (shoulders in desk 2)."""
    if frame[0, 0, 0] == 0:
        people = [_person(800), _person(100), _person(500, head_y=50)]  # desk 3, desk 1, no ROI
    else:
        p = _person(500)
        p[KP["head"], 2] = 0.1  # head masked -> (0, 0); must NOT fall into desk 1 at x = 0
        people = [p]
    return detections_from_array(np.stack(people))


def _frames():
    for i in range(2):
        yield i, np.full((4, 4, 3), i, dtype=np.uint8)


def test_columns_match_synthetic_schema():
    synth = generate_frame_table("hand_raise", n_videos=1, n_desks=1, n_frames=2)
    assert frame_table_columns() == [c for c in synth.columns if c != "hand_raise"]
    conf_cols = frame_table_columns(include_conf=True)[len(frame_table_columns()) :]
    assert conf_cols == [f"kp_{k}_conf" for k in range(NUM_KEYPOINTS)]


def test_extract_assigns_desks_with_camera_layout(layout):
    table = extract_frame_table(_frames(), _fake_pose, layout, video="v0")
    assert list(table.columns) == frame_table_columns()
    assert table[["frame", "desk_no"]].values.tolist() == [[0, 1], [0, 3], [1, 2]]
    assert (table["video"] == "v0").all()
    desk3 = table[(table.frame == 0) & (table.desk_no == 3)].iloc[0]
    assert desk3["kp_0_x"] == 800 and desk3["kp_9_y"] == 390
    assert desk3[["position_a", "position_b", "position_c", "position_d"]].tolist() == [
        0.0, 0.5, 1.0, 0.0
    ]
    missing_head = table[table.frame == 1].iloc[0]
    assert missing_head["kp_0_x"] == 0 and missing_head["kp_0_y"] == 0


def test_extract_with_conf_columns(layout):
    table = extract_frame_table(_frames(), _fake_pose, layout, video="v0", include_conf=True)
    assert list(table.columns) == frame_table_columns(include_conf=True)
    assert table.loc[table.frame == 1, "kp_0_conf"].iloc[0] == pytest.approx(0.1)


def test_duplicate_people_in_one_desk(layout):
    det = detections_from_array(np.stack([_person(100, conf=0.6), _person(150, conf=0.95)]))
    best = rows_for_frame("v", 0, det, layout)
    assert len(best) == 1 and best[0]["kp_0_x"] == 150
    both = rows_for_frame("v", 0, det, layout, duplicates="all")
    assert [r["kp_0_x"] for r in both] == [150, 100]
    with pytest.raises(ValueError):
        rows_for_frame("v", 0, det, layout, duplicates="first")


def test_empty_frames_give_empty_table_with_schema(layout):
    table = extract_frame_table(_frames(), lambda f: PoseDetections.empty(), layout, video="v")
    assert table.empty and list(table.columns) == frame_table_columns()


def test_video_reading_requires_opencv(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(ImportError, match="OpenCV"):
        next(iter_video_frames(tmp_path / "missing.mp4"))
    with pytest.raises(ValueError):
        next(iter_video_frames(tmp_path / "missing.mp4", stride=0))
