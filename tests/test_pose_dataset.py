import numpy as np
import pytest
import yaml

from qiyas_akd.keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS
from qiyas_akd.pose.dataset import (
    ANNOTATION_ORDER,
    FLIP_IDX,
    build_data_yaml,
    compute_flip_idx,
    plan_video_split,
    reorder_label_dir,
    reorder_label_line,
    reorder_permutation,
    split_by_video,
    video_key,
    write_data_yaml,
)


def test_flip_idx_is_involution_matching_left_right_names():
    assert len(FLIP_IDX) == NUM_KEYPOINTS
    assert sorted(FLIP_IDX) == list(range(NUM_KEYPOINTS))
    assert all(FLIP_IDX[FLIP_IDX[i]] == i for i in range(NUM_KEYPOINTS))
    for i, j in enumerate(FLIP_IDX):
        a, b = KEYPOINT_NAMES[i], KEYPOINT_NAMES[j]
        if a.startswith("left_"):
            assert b == "right_" + a[5:]
        elif a.startswith("right_"):
            assert b == "left_" + a[6:]
        else:
            assert i == j  # head, neck are on the midline
    assert list(FLIP_IDX) == [0, 1, 3, 2, 7, 8, 9, 4, 5, 6]


def test_compute_flip_idx_rejects_unmatched_side():
    with pytest.raises(ValueError):
        compute_flip_idx(["head", "left_hand"])


def test_data_yaml_content(tmp_path):
    d = build_data_yaml(tmp_path)
    assert d["kpt_shape"] == [10, 3]
    assert d["names"] == {0: "person"}
    assert d["flip_idx"] == list(FLIP_IDX)
    assert d["train"] == "train/images" and d["val"] == "val/images"
    assert build_data_yaml(tmp_path, flip="original")["flip_idx"] == []
    with pytest.raises(ValueError):
        build_data_yaml(tmp_path, flip="mirror")

    out = write_data_yaml(tmp_path / "cfg" / "data.yaml", tmp_path)
    text = out.read_text(encoding="utf8")
    assert "#   0: head" in text and "#   9: right_hand" in text
    assert yaml.safe_load(text) == d
    orig = write_data_yaml(tmp_path / "orig.yaml", tmp_path, flip="original").read_text("utf8")
    assert "disable horizontal-flip" in orig


def _label_line(order):
    """Label whose keypoint for name n has x = index of n in the schema / 100."""
    pts = " ".join(f"{KEYPOINT_NAMES.index(n) / 100:.6f} 0.500000 2" for n in order)
    return f"0 0.500000 0.400000 0.200000 0.300000 {pts}"


def test_reorder_label_line_annotation_to_schema():
    out = reorder_label_line(_label_line(ANNOTATION_ORDER)).split()
    assert out[:5] == ["0", "0.500000", "0.400000", "0.200000", "0.300000"]
    xs = [float(out[5 + 3 * k]) for k in range(NUM_KEYPOINTS)]
    assert np.allclose(xs, np.arange(NUM_KEYPOINTS) / 100)
    assert all(out[7 + 3 * k] == "2" for k in range(NUM_KEYPOINTS))
    # already in schema order -> identity
    assert reorder_label_line(_label_line(KEYPOINT_NAMES), KEYPOINT_NAMES) == _label_line(
        KEYPOINT_NAMES
    )
    assert reorder_label_line("   ") == ""
    with pytest.raises(ValueError):
        reorder_label_line("0 0.5 0.5 0.1 0.1 0.2 0.2 2")


def test_reorder_permutation_validates():
    assert reorder_permutation(KEYPOINT_NAMES) == list(range(NUM_KEYPOINTS))
    with pytest.raises(ValueError):
        reorder_permutation(KEYPOINT_NAMES[:-1])


def test_reorder_label_dir(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "a.txt").write_text(_label_line(ANNOTATION_ORDER) + "\n\n", encoding="utf8")
    (src / "bad.txt").write_text("0 1 2\n", encoding="utf8")
    with pytest.raises(ValueError, match="bad.txt:1"):
        reorder_label_dir(src, dst)
    (src / "bad.txt").unlink()
    assert reorder_label_dir(src, dst) == 1
    lines = (dst / "a.txt").read_text(encoding="utf8").splitlines()
    assert lines == [reorder_label_line(_label_line(ANNOTATION_ORDER))]
    with pytest.raises(ValueError, match="idempotent"):
        reorder_label_dir(src, src)


@pytest.mark.parametrize(
    ("name", "video"),
    [
        ("c2_v4_f0001.jpg", "c2_v4"),
        ("c13-1_f0003.png", "c13-1"),
        ("clip7_frame_0042.jpg", "clip7"),
        ("clip7_0042.jpg", "clip7"),
        ("video_12_0001.jpg", "video_12"),
    ],
)
def test_video_key(name, video):
    assert video_key(name) == video


def test_video_key_rejects_unparseable_name():
    with pytest.raises(ValueError, match="named group"):
        video_key("noframeindex.jpg")


def test_plan_video_split_keeps_each_video_on_one_side():
    names = [f"vid{v}_f{f:04d}.jpg" for v in range(10) for f in range(7)]
    plan = plan_video_split(names, val_fraction=0.3, seed=1)
    assert set(plan) == set(names)
    side_of: dict[str, set[str]] = {}
    for n, side in plan.items():
        side_of.setdefault(video_key(n), set()).add(side)
    assert all(len(s) == 1 for s in side_of.values())
    assert sum("val" in s for s in side_of.values()) == 3
    assert plan == plan_video_split(reversed(names), val_fraction=0.3, seed=1)  # deterministic


def test_split_by_video_copies_files(tmp_path):
    img, lab, out = tmp_path / "images", tmp_path / "labels", tmp_path / "out"
    img.mkdir()
    lab.mkdir()
    for v in range(5):
        for f in range(3):
            stem = f"v{v}_f{f:04d}"
            (img / f"{stem}.jpg").write_bytes(b"x")
            if not (v == 0 and f == 0):
                (lab / f"{stem}.txt").write_text("0", encoding="utf8")
    (img / "notes.md").write_text("ignored", encoding="utf8")
    s = split_by_video(img, lab, out, val_fraction=0.4, seed=0)
    assert s.n_train_images + s.n_val_images == 15
    assert len(s.val_videos) == 2 and not set(s.train_videos) & set(s.val_videos)
    assert s.missing_labels == ["v0_f0000.jpg"]
    for side, vids in (("train", s.train_videos), ("val", s.val_videos)):
        found = {video_key(p.name) for p in (out / side / "images").iterdir()}
        assert found == set(vids)
        n_labels = len(list((out / side / "labels").iterdir()))
        assert n_labels == 3 * len(vids) - ("v0" in vids)
    assert len(list(img.glob("*.jpg"))) == 15  # copied, source untouched
