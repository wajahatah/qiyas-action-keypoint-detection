import math

import numpy as np
import pytest

from qiyas_akd.desks import CameraLayout, Desk
from qiyas_akd.keypoints import KP, NUM_KEYPOINTS
from qiyas_akd.looking_around import (
    DEFAULT_POLY_COEFFS,
    ConstantThreshold,
    FieldOfViewThreshold,
    LinearThreshold,
    LookingAroundConfig,
    PolynomialThreshold,
    centre_angle,
    decide,
    depth_proxy,
    detect_sequence,
    detect_table,
    ear_line_angle,
    evaluate_table,
    field_of_view,
    head_in_visual_field,
    make_threshold,
    perpendicular_foot,
    persistence_filter,
    point_in_triangle,
    synthetic_layout,
)
from qiyas_akd.looking_around.thresholds import ears_outside_band
from qiyas_akd.synthetic import generate_frame_table

DESK = Desk(
    desk_id=1,
    roi=(0, 150, 400, 720),
    ref_line=(0, 300, 400, 300),
    base_angle_in=75,
    base_angle_out=90,
)


def person(head=(200, 400), left=(180, 410), right=(220, 410)) -> np.ndarray:
    xy = np.full((NUM_KEYPOINTS, 2), 50.0)
    xy[KP["head"]] = head
    xy[KP["left_ear"]] = left
    xy[KP["right_ear"]] = right
    return xy


# ---------------------------------------------------------------- geometry


def test_ear_line_angle_level_and_turned():
    assert ear_line_angle((100, 200), (140, 200)) == pytest.approx(0.0)
    assert ear_line_angle((100, 200), (140, 240)) == pytest.approx(45.0)  # right ear lower
    assert ear_line_angle((100, 200), (140, 160)) == pytest.approx(-45.0)
    # vertical ear line is an extreme turn, not 0° (the original's x1 == x2 guard returned 0)
    assert abs(ear_line_angle((100, 200), (100, 240))) == pytest.approx(90.0)
    # ears crossed over (head turned away) → near ±180°
    assert abs(ear_line_angle((140, 200), (100, 202))) > 170


def test_missing_ear_gives_no_angle():
    assert ear_line_angle((0, 0), (140, 200)) is None
    assert ear_line_angle((100, 200), (np.nan, 200)) is None
    assert centre_angle((120, 150), (100, 200), (0, 0)) is None
    assert head_in_visual_field((120, 150), (0, 0), (140, 200), 45) is None


def test_centre_angle_convention():
    ears = ((100, 200), (140, 200))
    assert centre_angle((120, 150), *ears) == pytest.approx(0.0)  # head straight up
    assert centre_angle((170, 150), *ears) == pytest.approx(45.0)  # head to the right
    assert centre_angle((70, 150), (100, 200), (140, 200)) == pytest.approx(-45.0)
    assert abs(centre_angle((120, 250), (100, 200), (140, 200))) == pytest.approx(180.0)


def test_perpendicular_foot():
    # same example as the original vision_direction sketch: A(1,0), B(4,5), C(2,5)
    d = perpendicular_foot((2, 5), (1, 0), (4, 5))
    ab = np.subtract((4, 5), (1, 0))
    cd = np.subtract(d, (2, 5))
    assert abs(np.dot(ab, cd)) < 1e-9  # CD ⟂ AB
    assert d == pytest.approx((1 + 3 * 28 / 34, 5 * 28 / 34))
    with pytest.raises(ValueError):
        perpendicular_foot((0, 1), (2, 2), (2, 2))


def test_point_in_triangle():
    a, b, c = (0.0, 0.0), (10.0, 0.0), (0.0, 10.0)
    assert point_in_triangle((2.0, 2.0), a, b, c)
    assert point_in_triangle((5.0, 5.0), a, b, c)  # on the hypotenuse
    assert not point_in_triangle((6.0, 6.0), a, b, c)
    assert not point_in_triangle((-0.1, 1.0), a, b, c)


def test_visual_field_triangle_inside_outside():
    left, right = (100, 200), (140, 200)
    assert head_in_visual_field((120, 170), left, right, half_angle_deg=45) is True  # upright
    # the apex is the perpendicular foot, so a head shifted along a level ear line stays inside
    assert head_in_visual_field((135, 170), left, right, half_angle_deg=45) is True
    assert head_in_visual_field((120, 230), left, right, half_angle_deg=45) is False  # below
    assert head_in_visual_field((120, 0), left, right, half_angle_deg=45) is False  # beyond cone
    tilted_left, tilted_right = (110, 180), (130, 240)  # ear line ~72° → head off to the side
    assert head_in_visual_field((150, 200), tilted_left, tilted_right, half_angle_deg=45) is False
    # tilting the whole head (ears + head rotated together) keeps it inside a wide cone
    ang = math.radians(20)
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    c = np.array([120, 200])
    pts = [c + rot @ (np.array(p) - c) for p in [(120, 170), left, right]]
    assert head_in_visual_field(pts[0], pts[1], pts[2], half_angle_deg=30) is True
    assert head_in_visual_field(pts[0], pts[1], pts[2], half_angle_deg=10) is False


def test_field_of_view():
    assert field_of_view(0) == pytest.approx(180.0)
    assert field_of_view(100, k=100) == pytest.approx(90.0)
    assert field_of_view(50) > field_of_view(200)
    with pytest.raises(ValueError):
        field_of_view(-1)
    with pytest.raises(ValueError):
        field_of_view(10, k=0)


# ---------------------------------------------------------------- thresholds


def test_depth_proxy_clipped():
    assert depth_proxy(300, DESK) == 0.0
    assert depth_proxy(720, DESK) == 1.0
    assert depth_proxy(510, DESK) == pytest.approx(0.5)
    assert depth_proxy(100, DESK) == 0.0 and depth_proxy(900, DESK) == 1.0
    assert depth_proxy(100, DESK, clip=False) < 0
    bad = Desk(desk_id=9, roi=(0, 0, 10, 100), ref_line=(0, 200, 10, 200))
    with pytest.raises(ValueError):
        depth_proxy(50, bad)


def test_band_rule_uses_left_ear_and_either_ref_end():
    tilted = Desk(desk_id=2, roi=(0, 150, 400, 720), ref_line=(0, 280, 400, 320))
    assert ears_outside_band((10, 300), tilted)  # above the right end (320)
    assert not ears_outside_band((10, 330), tilted)
    no_line = Desk(desk_id=3, roi=(0, 150, 400, 720))
    assert ears_outside_band((10, 100), no_line) and not ears_outside_band((10, 200), no_line)


def test_linear_threshold_base_and_monotonic_in_depth():
    thr = LinearThreshold(k=15)
    # above the reference line → outside → base_angle_out, depth clipped to 0
    assert thr((180, 250), (220, 250), DESK) == 90
    # below → inside → base_angle_in − int(x·k)
    ys = np.linspace(301, 720, 30)
    values = [thr((180, y), (220, y), DESK) for y in ys]
    assert values[0] == 75 and values[-1] == 60
    assert np.all(np.diff(values) <= 0)  # never increases with depth
    assert all(float(v).is_integer() for v in values)  # integer steps like the original
    smooth = LinearThreshold(k=30, integer_step=False)
    assert smooth((180, 510), (220, 510), DESK) == pytest.approx(75 - 15)
    assert LinearThreshold(k=30, base=60)((180, 720), (220, 720), DESK) == 30


def test_polynomial_threshold_configurable():
    poly = PolynomialThreshold()
    assert poly.coeffs == DEFAULT_POLY_COEFFS
    assert poly((180, 300), (220, 300), DESK) == pytest.approx(50.668)
    assert poly((180, 720), (220, 720), DESK) == pytest.approx(sum(DEFAULT_POLY_COEFFS))
    xs = np.linspace(301, 720, 50)
    vals = [poly((180, y), (220, y), DESK) for y in xs]
    assert vals[0] > vals[-1]
    linear_as_poly = PolynomialThreshold(coeffs=(-10.0, 70.0))
    assert linear_as_poly((180, 510), (220, 510), DESK) == pytest.approx(65.0)
    with pytest.raises(ValueError):
        PolynomialThreshold(coeffs=())


def test_make_threshold():
    assert isinstance(make_threshold(None), LinearThreshold)
    assert make_threshold({"kind": "linear", "k": 30}).k == 30
    assert make_threshold({"kind": "polynomial", "coeffs": [1, 2]}).coeffs == (1.0, 2.0)
    assert isinstance(make_threshold({"kind": "constant", "angle": 45}), ConstantThreshold)
    assert isinstance(make_threshold({"kind": "fov", "k": 50}), FieldOfViewThreshold)
    with pytest.raises(ValueError):
        make_threshold({"kind": "nope"})


# ---------------------------------------------------------------- decisions


def test_decide_ear_line_flags():
    thr = ConstantThreshold(60)
    assert decide(person(), DESK, "ear_line", thr) is False  # ears level
    turned = person(left=(190, 380), right=(210, 440))  # ~72°
    assert decide(turned, DESK, "ear_line", thr) is True
    assert decide(person(left=(0, 0)), DESK, "ear_line", thr) is None  # missing ear: no decision


def test_decide_centre_and_triangle():
    thr = ConstantThreshold(45)
    upright = person(head=(200, 380), left=(180, 410), right=(220, 410))
    head_aside = person(head=(240, 390), left=(180, 410), right=(220, 410))  # ~63° from upright
    ears_turned = person(head=(200, 380), left=(190, 380), right=(210, 440))
    assert decide(upright, DESK, "centre_angle", thr) is False
    assert decide(head_aside, DESK, "centre_angle", thr) is True
    assert decide(upright, DESK, "triangle", thr) is False
    assert decide(ears_turned, DESK, "triangle", thr) is True
    for method in ("centre_angle", "triangle"):
        assert decide(person(head=(0, 0)), DESK, method, thr) is None
    with pytest.raises(ValueError):
        decide(upright, DESK, "bogus", thr)
    with pytest.raises(ValueError):
        decide(upright[:5], DESK, "ear_line", thr)


def test_persistence_filter_suppresses_blips():
    raw = [0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0]
    out = persistence_filter(raw, window=5, min_hits=4)
    assert out.tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    assert persistence_filter([0, 1, 0, 1, 0], window=3, min_hits=1).tolist() == [0, 1, 0, 1, 0]
    lingering = persistence_filter([1, 1, 1, 0, 0], window=3, min_hits=2, require_current=False)
    assert lingering.tolist() == [0, 1, 1, 1, 0]
    with pytest.raises(ValueError):
        persistence_filter(raw, window=3, min_hits=4)
    with pytest.raises(ValueError):
        LookingAroundConfig(window=3, min_hits=0)
    with pytest.raises(ValueError):
        LookingAroundConfig(method="nope")


def test_detect_sequence_single_frame_blip_is_not_an_alert():
    level = person()
    turned = person(left=(190, 380), right=(210, 440))
    seq = np.stack([level] * 5 + [turned] + [level] * 5 + [turned] * 8 + [level] * 3)
    cfg = LookingAroundConfig(threshold=ConstantThreshold(60), window=5, min_hits=4)
    flags = detect_sequence(seq, DESK, cfg)
    assert flags[:11].sum() == 0  # the isolated frame 5 is suppressed
    assert flags[11:19].sum() == 5 and flags[19:].sum() == 0


def test_synthetic_layout_assigns_synthetic_heads():
    df = generate_frame_table("looking_around", n_videos=1, n_desks=4, n_frames=10, seed=1)
    layout = synthetic_layout(4)
    assert isinstance(layout, CameraLayout) and len(layout.desks) == 4
    for row in df.itertuples():
        xy = np.array([[getattr(row, f"kp_{k}_x"), getattr(row, f"kp_{k}_y")] for k in range(10)])
        if not (xy[KP["head"]] == 0).all():
            assert layout.assign(xy) == row.desk_no


@pytest.mark.parametrize("method", ["ear_line", "triangle"])
def test_synthetic_table_recall(method):
    df = generate_frame_table("looking_around", n_videos=3, n_desks=4, n_frames=150, seed=7)
    layout = synthetic_layout(4)
    cfg = LookingAroundConfig(method=method)
    counts = evaluate_table(df, layout, cfg)
    assert counts.tp + counts.fn > 0
    assert counts.recall > 0
    if method == "ear_line":
        # FPs here are fragments of true events: a dropped ear is "no decision" → negative (as in
        # the original, which reset the per-desk flag every frame), splitting one alert in two.
        assert counts.recall > 0.8 and counts.precision > 0.6
    out = detect_table(df, layout, cfg)
    assert {"la_raw", "la_flag"} <= set(out.columns) and len(out) == len(df)
    with pytest.raises(KeyError):
        evaluate_table(df, layout, cfg, label_column="missing")


def test_config_files_and_cli(capsys):
    import json
    from pathlib import Path

    import yaml

    from qiyas_akd.looking_around.evaluate import main

    cfg_dir = Path(__file__).resolve().parents[1] / "configs" / "looking_around"
    files = sorted(cfg_dir.glob("*.yaml"))
    assert files
    for f in files:
        cfg = LookingAroundConfig.from_dict(yaml.safe_load(f.read_text(encoding="utf8")))
        assert cfg.method in ("ear_line", "centre_angle", "triangle")
    poly = LookingAroundConfig.from_dict(
        yaml.safe_load((cfg_dir / "ear_line_polynomial.yaml").read_text(encoding="utf8"))
    )
    assert isinstance(poly.threshold, PolynomialThreshold)
    with pytest.raises(ValueError):
        LookingAroundConfig.from_dict({"methd": "ear_line"})
    assert main(["--synthetic", "--n-videos", "2", "--n-frames", "120"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["method"] == "ear_line" and out["recall"] > 0
