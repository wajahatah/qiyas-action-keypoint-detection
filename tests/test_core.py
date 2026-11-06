import numpy as np
import pytest

from qiyas_akd.evaluation import Counts, events_from_frames, grouped_split, match_events
from qiyas_akd.keypoints import (
    MISSING,
    NUM_KEYPOINTS,
    SKELETON,
    bone_lengths,
    kp_columns,
    mask_low_confidence,
    normalize_xy,
    roi_relative,
)
from qiyas_akd.synthetic import BEHAVIOURS, generate_frame_table


def test_schema():
    assert NUM_KEYPOINTS == 10
    assert all(0 <= a < 10 and 0 <= b < 10 for a, b in SKELETON)


def test_normalize_marks_missing_and_rounds():
    xy = np.array([[[640.0, 360.0], [0.0, 0.0], [np.nan, 5.0], [1279.0, 719.0]]])
    out = normalize_xy(xy)
    assert out[0, 0].tolist() == [0.5, 0.5]
    assert out[0, 1].tolist() == [MISSING, MISSING]
    assert out[0, 2].tolist() == [MISSING, MISSING]
    assert out[0, 3].tolist() == [round(1279 / 1280, 3), round(719 / 720, 3)]


def test_mask_low_confidence():
    xy = np.ones((3, 2)) * 10
    out = mask_low_confidence(xy, np.array([0.9, 0.49, 0.5]))
    assert out[1].tolist() == [0, 0] and out[0].tolist() == [10, 10] and out[2].tolist() == [10, 10]
    with pytest.raises(ValueError):
        mask_low_confidence(xy, np.ones(2))


def test_roi_relative():
    xy = np.array([[110.0, 220.0], [0.0, 0.0]])
    out = roi_relative(xy, (100, 200, 200, 300))
    assert np.allclose(out[0], [0.1, 0.2]) and out[1].tolist() == [MISSING, MISSING]
    with pytest.raises(ValueError):
        roi_relative(xy, (10, 10, 5, 20))


def test_bone_lengths_missing_endpoint_is_zero():
    xy = np.zeros((NUM_KEYPOINTS, 2))
    xy[0] = (0, 3)  # head... but (0,3) is not missing (not both zero)
    xy[1] = (4, 3)
    d = bone_lengths(xy)
    assert d.shape == (len(SKELETON),)
    assert d[0] == pytest.approx(4.0)  # head-neck
    assert d[1] == 0.0  # head-left_ear: ear missing


def test_kp_columns_layouts():
    assert len(kp_columns()) == 20
    km = kp_columns(3, "kp_major")
    assert km[:4] == ["kp_0_x_t0", "kp_0_x_t1", "kp_0_x_t2", "kp_0_y_t0"]
    tm = kp_columns(3, "time_major")
    assert tm[:3] == ["kp_0_x_t0", "kp_0_y_t0", "kp_1_x_t0"]
    assert sorted(km) == sorted(tm) and len(km) == 60
    with pytest.raises(ValueError):
        kp_columns(3, "bogus")


def test_counts_metrics():
    c = Counts(8, 2, 2)
    assert c.precision == 0.8 and c.recall == 0.8 and c.f1 == pytest.approx(0.8)
    assert Counts(0, 0, 0).f1 == 0.0
    assert (c + Counts(1, 1, 1)) == Counts(9, 3, 3)


def test_events_and_matching():
    pred = [0, 1, 1, 1, 0, 1, 0, 0, 1, 1]
    assert events_from_frames(pred) == [(1, 3), (5, 5), (8, 9)]
    assert events_from_frames(pred, min_len=2) == [(1, 3), (8, 9)]
    c = match_events([(1, 3), (8, 9), (20, 22)], [(2, 4), (12, 14)])
    assert c == Counts(tp=1, fp=2, fn=1)
    assert match_events([(5, 6)], [(8, 9)], tolerance=2) == Counts(1, 0, 0)


def test_grouped_split_no_leakage():
    groups = np.repeat([f"v{i}" for i in range(10)], 7)
    tr, te = grouped_split(groups, 0.3, seed=1)
    assert set(groups[tr]).isdisjoint(set(groups[te]))
    assert len(tr) + len(te) == len(groups)
    assert np.array_equal(te, grouped_split(groups, 0.3, seed=1)[1])
    with pytest.raises(ValueError):
        grouped_split(["a"] * 5)


@pytest.mark.parametrize("behaviour", BEHAVIOURS)
def test_synthetic_table(behaviour):
    df = generate_frame_table(behaviour, n_videos=2, n_desks=2, n_frames=60, seed=3)
    assert len(df) == 2 * 2 * 60
    assert {"video", "frame", "desk_no", behaviour, "kp_9_y", "position_d"} <= set(df.columns)
    assert df[behaviour].isin([0, 1]).all()
    assert df[behaviour].sum() > 0
