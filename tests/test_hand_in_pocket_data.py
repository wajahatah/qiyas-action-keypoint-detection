"""Hand in pocket — dataset preparation: windowing, labelling, skipping, preprocessing, balancing,
augmentation. All tables are hand-built so every expected value can be checked by eye."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qiyas_akd.desks import CameraLayout
from qiyas_akd.hand_in_pocket.dataprep.augment import jitter
from qiyas_akd.hand_in_pocket.dataprep.balancing import (
    block_balance,
    positive_segments,
    segment_balance,
)
from qiyas_akd.hand_in_pocket.dataprep.pipeline import DataConfig
from qiyas_akd.hand_in_pocket.dataprep.preprocess import (
    BONE_COLS,
    add_bone_lengths,
    add_position_features,
    apply_confidence_mask,
    kp_pairs,
    normalise_points,
    normalise_table,
)
from qiyas_akd.hand_in_pocket.dataprep.windowing import (
    WindowSpec,
    make_windows,
    window_label,
    windowed_feature_columns,
)
from qiyas_akd.keypoints import MISSING, NUM_KEYPOINTS, kp_columns, normalize_xy

ROOT = Path(__file__).resolve().parents[1]
LABEL = "hand_in_pocket"


def track(labels, video="v0", desk=1, start=0, frames=None):
    """One desk track; kp_k_x = 100 * (k + 1) + frame, kp_k_y = 50 * (k + 1) + frame (pixels)."""
    frames = list(range(start, start + len(labels))) if frames is None else list(frames)
    rows = []
    for f, lab in zip(frames, labels, strict=True):
        row = {"video": video, "frame": f, "desk_no": desk}
        for k in range(NUM_KEYPOINTS):
            row[f"kp_{k}_x"] = 100.0 * (k + 1) + f
            row[f"kp_{k}_y"] = 50.0 * (k + 1) + f
        row.update(position_a=1.0, position_b=0.0, position_c=0.5, position_d=0.0)
        row[LABEL] = lab
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- windowing


def test_window_shapes_and_kp_major_order():
    w = make_windows(track([0] * 8), WindowSpec(window=5))
    assert len(w) == 4
    feats = windowed_feature_columns(WindowSpec(window=5))
    assert feats == kp_columns(5, "kp_major")
    assert [c for c in w.columns if c.startswith("kp_")] == feats
    assert list(w.columns[-6:]) == [
        "position_a",
        "position_b",
        "position_c",
        "position_d",
        "label_fraction",
        LABEL,
    ]
    row = w.iloc[1]  # window over frames 1..5
    assert row["frame"] == 1 and row["frame_end"] == 5
    assert row["kp_0_x_t3"] == 100 + 4 and row["kp_9_y_t0"] == 500 + 1
    assert row["frame_t4"] == 5


def test_time_major_layout():
    w = make_windows(track([0] * 6), WindowSpec(window=3, layout="time_major"))
    assert [c for c in w.columns if c.startswith("kp_")] == kp_columns(3, "time_major")


@pytest.mark.parametrize(
    ("rule", "param", "expected"),
    [
        ("any", None, [1, 1, 1, 1, 1, 1]),
        ("all", None, [0, 0, 0, 0, 0, 0]),
        ("fraction", 0.5, [0, 1, 1, 1, 0, 0]),
        ("min_count", 2, [1, 1, 1, 1, 1, 0]),
        ("center", None, [0, 1, 1, 1, 0, 0]),
    ],
)
def test_label_rules(rule, param, expected):
    labels = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  # windows of 5 starting at 0..5
    w = make_windows(track(labels), WindowSpec(window=5, label_rule=rule, label_param=param))
    assert w[LABEL].tolist() == expected


def test_window_label_function_and_validation():
    lab = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    assert window_label(lab, "min_count", 3).tolist() == [1, 1]
    assert window_label(lab, "fraction", 0.7).tolist() == [0, 1]
    with pytest.raises(ValueError):
        WindowSpec(label_rule="fraction")
    with pytest.raises(ValueError):
        WindowSpec(label_rule="bogus")
    with pytest.raises(ValueError):
        WindowSpec(window=5, keep=(0, 5))
    with pytest.raises(ValueError):
        WindowSpec(window=5, keep=(0, 2), random_drop=1)


def test_frame_skipping_t3_has_64_inputs_and_full_window_label():
    spec = WindowSpec(window=5, keep=(0, 2, 4))
    cfg = DataConfig(windowing=spec)
    assert len(cfg.feature_columns) == 64
    assert cfg.feature_columns[:3] == ["kp_0_x_t0", "kp_0_x_t2", "kp_0_x_t4"]
    labels = [0, 1, 0, 0, 0, 0, 0]  # only frame 1 positive → t1 of window 0, which is dropped
    w = make_windows(track(labels), spec)
    assert len(w) == 3
    assert w.iloc[0]["kp_0_x_t2"] == 100 + 2 and w.iloc[0]["kp_0_x_t4"] == 100 + 4
    assert "kp_0_x_t1" not in w.columns
    assert w[LABEL].tolist() == [1, 1, 0]  # label computed on all 5 frames


def test_row_skipping_and_frame_step():
    w = make_windows(track([0] * 10), WindowSpec(window=3, row_step=2))
    assert w["frame"].tolist() == [0, 2, 4, 6]
    w = make_windows(track([0] * 10), WindowSpec(window=3, frame_step=2))
    assert w["frame"].tolist() == [0, 2, 4]
    assert w.iloc[0]["frame_t1"] == 2 and w.iloc[0]["kp_0_x_t2"] == 100 + 4


def test_drop_mixed_removes_transition_windows():
    labels = [0, 0, 0, 1, 1, 1, 1, 0]
    w = make_windows(track(labels), WindowSpec(window=3, drop_mixed=True))
    assert w["label_fraction"].isin([0.0, 1.0]).all()
    assert w["frame"].tolist() == [0, 3, 4]


def test_random_drop_keeps_order_and_renumbers_steps():
    w = make_windows(track([0] * 12), WindowSpec(window=5, random_drop=1, seed=3))
    names = [c for c in w.columns if c.startswith("kp_0_x_")]
    assert names == ["kp_0_x_t0", "kp_0_x_t1", "kp_0_x_t2", "kp_0_x_t3"]
    vals = w[names].to_numpy()
    assert (np.diff(vals, axis=1) > 0).all()  # frames stay in temporal order
    assert ((vals[:, -1] - vals[:, 0]) <= 4).all()
    again = make_windows(track([0] * 12), WindowSpec(window=5, random_drop=1, seed=3))
    pd.testing.assert_frame_equal(w, again)


def test_grouping_gaps_short_tracks_and_chunks():
    df = pd.concat(
        [
            track([0] * 6, video="v0", desk=1),
            track([1] * 6, video="v0", desk=2),
            track([0] * 2, video="v1", desk=1),  # shorter than window → skipped
            track([0] * 8, video="v2", desk=1, frames=[0, 1, 2, 3, 10, 11, 12, 13]),
        ]
    )
    w = make_windows(df, WindowSpec(window=3))
    assert set(map(tuple, w[["video", "desk_no"]].drop_duplicates().to_numpy())) == {
        ("v0", 1),
        ("v0", 2),
        ("v2", 1),
    }
    assert w[w["desk_no"] == 2][LABEL].eq(1).all()
    gap = make_windows(df, WindowSpec(window=3, max_frame_gap=1))
    assert len(gap[gap["video"] == "v2"]) == 4  # 2 windows per 4-frame run, none across the gap
    chunked = df.assign(chunk=np.r_[[0] * 14, [0, 0, 0, 0, 1, 1, 1, 1]])
    assert len(make_windows(chunked, WindowSpec(window=3)).query("video == 'v2'")) == 4


# ---------------------------------------------------------------- preprocessing


def test_normalise_points_frame_truncates_then_matches_normalize_xy():
    xy = np.array([[639.9, 360.7], [0.0, 0.0], [1279.4, 1.2]])
    out = normalise_points(xy, "frame")
    np.testing.assert_array_equal(out, normalize_xy(np.trunc(xy)))
    assert out[0].tolist() == pytest.approx([round(639 / 1280, 3), 0.5])
    assert out[1].tolist() == [MISSING, MISSING]
    raw = normalise_points(xy, "none")
    assert raw[0].tolist() == [639.0, 360.0] and raw[1].tolist() == [0.0, 0.0]


def test_normalise_points_roi_and_errors():
    out = normalise_points(np.array([[150.0, 250.0], [0.0, 0.0]]), "roi", roi=(100, 200, 200, 300))
    assert out[0].tolist() == [0.5, 0.5] and out[1].tolist() == [MISSING, MISSING]
    with pytest.raises(ValueError):
        normalise_points(np.zeros((1, 2)), "roi")
    with pytest.raises(ValueError):
        normalise_points(np.zeros((1, 2)), "bogus")


def test_normalise_table_window_columns_and_roi_layout():
    w = make_windows(track([0] * 5), WindowSpec(window=3, keep=(0, 2)))
    n = normalise_table(w)
    assert n.loc[0, "kp_0_x_t2"] == pytest.approx(round(102 / 1280, 3))
    assert n.loc[0, "position_c"] == 0.5 and n.loc[0, "frame"] == 0  # non-kp untouched
    assert len(kp_pairs(list(w.columns))) == 20
    layout = CameraLayout.from_yaml(ROOT / "configs/cameras/example_camera.yaml")
    r = normalise_table(w, "roi", layout)  # desk 1 roi = (0, 150, 360, 720)
    assert r.loc[0, "kp_0_x_t0"] == pytest.approx(round(100 / 360, 3))
    assert r.loc[0, "kp_0_y_t0"] == pytest.approx(round((50 - 150) / 570, 3))


def test_confidence_mask_bone_lengths_and_positions():
    df = track([0, 1])
    df["kp_6_conf"] = [0.9, 0.2]
    for k in range(NUM_KEYPOINTS):
        if k != 6:
            df[f"kp_{k}_conf"] = 0.9
    m = apply_confidence_mask(df)
    assert not any(c.endswith("_conf") for c in m.columns)
    assert m.loc[1, "kp_6_x"] == 0 and m.loc[1, "kp_6_y"] == 0 and m.loc[0, "kp_6_x"] == 700
    b = add_bone_lengths(m)
    assert list(b.columns[-len(BONE_COLS) :]) == list(BONE_COLS)
    assert b.loc[1, "dist_5_6"] == 0.0 and b.loc[0, "dist_5_6"] > 0
    layout = CameraLayout.from_yaml(ROOT / "configs/cameras/example_camera.yaml")
    raw = m.drop(columns=["position_a", "position_b", "position_c", "position_d"])
    raw.loc[1, "desk_no"] = 99  # unknown desk → row dropped
    pos = add_position_features(raw, layout)
    assert len(pos) == 1
    assert pos.loc[0, ["position_a", "position_b"]].tolist() == [1.0, 0.5]


# ---------------------------------------------------------------- balancing


def test_positive_segments():
    assert positive_segments([0, 1, 1, 0, 1, 0, 0, 1]) == [(1, 2), (4, 4), (7, 7)]
    assert positive_segments([0, 0]) == []


def test_segment_balance_keeps_all_positives_with_context():
    labels = [0] * 10 + [1] * 4 + [0] * 20
    df = pd.concat([track(labels, desk=1), track([0] * 50, desk=2)])
    out = segment_balance(df, LABEL, negative_cap=30)
    d1 = out[out["desk_no"] == 1]
    assert d1[LABEL].sum() == 4  # every positive kept
    assert d1["frame"].tolist() == list(range(8, 16))  # 4 positives + 2 context each side
    assert len(out[out["desk_no"] == 2]) == 30  # all-negative desk capped
    assert out["chunk"].eq(0).all()


def test_segment_balance_merges_overlapping_context_without_duplicates():
    labels = [0] * 5 + [1] * 4 + [0, 0] + [1] * 4 + [0] * 10
    out = segment_balance(track(labels), LABEL)
    assert out["frame"].is_unique
    assert out[LABEL].sum() == 8
    assert out["chunk"].nunique() == 1  # the two context windows touch → one chunk
    far = [0] * 5 + [1] * 2 + [0] * 20 + [1] * 2 + [0] * 5
    out = segment_balance(track(far), LABEL)
    assert out["chunk"].nunique() == 2
    w = make_windows(out, WindowSpec(window=3))  # windows never cross the two chunks
    assert (w["frame_end"] - w["frame"]).eq(2).all()


def test_segment_balance_video_filter_passes_others_through():
    df = pd.concat([track([0] * 40, video="a"), track([0] * 40, video="b")])
    out = segment_balance(df, LABEL, negative_cap=10, videos={"a"})
    assert (out["video"] == "a").sum() == 10 and (out["video"] == "b").sum() == 40


def test_block_balance_respects_budget_and_tracks():
    df = pd.concat([track([0] * 40, desk=1), track([1] * 40, desk=2)])
    w = make_windows(df, WindowSpec(window=3))
    out = block_balance(w, n_negative=20, n_positive=10, label_col=LABEL, block=5, seed=1)
    assert (out[LABEL] == 0).sum() <= 20 and (out[LABEL] == 1).sum() <= 10
    assert len(out) % 5 == 0
    for i in range(0, len(out), 5):
        assert out.iloc[i : i + 5]["desk_no"].nunique() == 1
    again = block_balance(w, 20, 10, LABEL, 5, seed=1)
    pd.testing.assert_frame_equal(out, again)


# ---------------------------------------------------------------- augmentation


def test_jitter_keeps_missing_points_missing_and_bounds_noise():
    w = make_windows(track([1] * 6), WindowSpec(window=3))
    w.loc[:, "kp_6_x_t1"] = 0.0
    w.loc[:, "kp_6_y_t1"] = 0.0
    w.loc[0, "kp_0_x_t0"] = 2.0  # near the border: must not become 0 (= missing)
    out = jitter(w, max_px=4, times_positive=2, seed=0)
    assert len(out) == 3 * len(w)
    assert out["original"].tolist() == [1] * len(w) + [0] * (2 * len(w))
    aug = out[out["original"] == 0]
    assert (aug["kp_6_x_t1"] == 0).all() and (aug["kp_6_y_t1"] == 0).all()
    pd.testing.assert_frame_equal(out[out["original"] == 1].drop(columns="original"), w)
    orig = pd.concat([w, w], ignore_index=True)
    cols = [c for c in w.columns if c.startswith("kp_") and c not in ("kp_6_x_t1", "kp_6_y_t1")]
    diff = aug[cols].to_numpy() - orig[cols].to_numpy()
    assert np.abs(diff).max() <= 4 and (aug[cols].to_numpy() >= 1).all()
    assert (diff != 0).any()


def test_jitter_class_counts_and_shared_row_noise():
    w = make_windows(track([0] * 4 + [1] * 4), WindowSpec(window=2, label_rule="all"))
    out = jitter(w, times_positive=3, times_negative=1, per_value=False, seed=2)
    n_pos, n_neg = int((w[LABEL] == 1).sum()), int((w[LABEL] == 0).sum())
    assert len(out) == len(w) + 3 * n_pos + n_neg
    aug = out[out["original"] == 0].reset_index(drop=True)
    src = pd.concat([w[w[LABEL] == 1]] * 3 + [w[w[LABEL] == 0]], ignore_index=True)
    cols = [c for c in w.columns if c.startswith("kp_")]
    d = aug[cols].to_numpy() - src[cols].to_numpy()
    assert (d == d[:, :1]).all()  # one shift per row for every coordinate
    assert len(jitter(w, times_positive=0)) == len(w)
