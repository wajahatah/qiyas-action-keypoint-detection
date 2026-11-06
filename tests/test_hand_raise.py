import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from qiyas_akd.desks import CameraLayout
from qiyas_akd.hand_raise import (
    RF_PARAM_GRID,
    KeypointMLP,
    TwoBranchMLP,
    build_features,
    clean_columns,
    elbow_angle,
    engineered_features,
    feature_names,
    generalised_features,
    grid_search_random_forest,
    hand_raised,
    load_tables,
    make_random_forest,
    multilabel_accuracy,
    normalised_features,
    parse_point_columns,
    predict_table,
    roi_left_from_layout,
    validate_binary_labels,
)
from qiyas_akd.hand_raise.train import TrainConfig, main, resolve_config, save_model, train
from qiyas_akd.keypoints import KEYPOINT_NAMES, KP, MISSING, NUM_KEYPOINTS
from qiyas_akd.synthetic import generate_frame_table

ROOT = Path(__file__).resolve().parents[1]
CAMERA = ROOT / "configs" / "cameras" / "example_camera.yaml"


def seated(cx=300.0, cy=300.0) -> np.ndarray:
    p = np.zeros((NUM_KEYPOINTS, 2))
    p[KP["head"]] = (cx, cy)
    p[KP["neck"]] = (cx, cy + 40)
    p[KP["left_ear"]] = (cx - 18, cy + 2)
    p[KP["right_ear"]] = (cx + 18, cy + 2)
    p[KP["left_shoulder"]] = (cx - 45, cy + 55)
    p[KP["right_shoulder"]] = (cx + 45, cy + 55)
    p[KP["left_elbow"]] = (cx - 60, cy + 105)
    p[KP["right_elbow"]] = (cx + 60, cy + 105)
    p[KP["left_hand"]] = (cx - 30, cy + 140)
    p[KP["right_hand"]] = (cx + 30, cy + 140)
    return p


def raised(side="right") -> np.ndarray:
    p = seated()
    sh = p[KP[f"{side}_shoulder"]]
    p[KP[f"{side}_elbow"]] = sh + (5, -55)
    p[KP[f"{side}_hand"]] = sh + (8, -115)
    return p


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return generate_frame_table("hand_raise", n_videos=6, n_desks=4, n_frames=80, seed=11)


# ---------------------------------------------------------------- rule


def test_elbow_angle():
    assert elbow_angle((0, 0.5), (0, 10), (0, 20)) == pytest.approx(180.0)
    assert elbow_angle((10, 10), (0, 10), (0, 20)) == pytest.approx(90.0)
    assert elbow_angle((0, 0), (0, 10), (0, 20)) is None  # missing shoulder
    assert elbow_angle((5, 10), (5, 10), (0, 20)) is None  # zero-length segment
    # confidence (3rd column) must not leak into the angle
    assert elbow_angle((10, 10, 0.2), (0, 10, 0.9), (0, 20, 0.5)) == pytest.approx(90.0)


def test_rule_on_constructed_poses():
    assert not hand_raised(seated())
    assert hand_raised(raised("right")) and hand_raised(raised("left"))
    bent_up = seated()
    bent_up[KP["right_elbow"]] = bent_up[KP["right_shoulder"]] + (60, 0)
    bent_up[KP["right_hand"]] = bent_up[KP["right_shoulder"]] + (60, -60)  # 90° elbow, hand up
    assert not hand_raised(bent_up)
    straight_down = seated()
    straight_down[KP["left_elbow"]] = straight_down[KP["left_shoulder"]] + (0, 50)
    straight_down[KP["left_hand"]] = straight_down[KP["left_shoulder"]] + (0, 100)
    assert not hand_raised(straight_down)  # straight arm but wrist below shoulder


def test_rule_requires_both_arms_like_original():
    p = raised("right")
    p[KP["left_hand"]] = (0, 0)  # other arm occluded
    assert not hand_raised(p)
    assert hand_raised(p, require_both_arms=False)


def test_rule_table(table):
    pred = predict_table(table)
    assert pred.shape == (len(table),) and set(np.unique(pred)) <= {0, 1}
    y = table["hand_raise"].to_numpy()
    assert pred[y == 1].mean() > 0.5 and pred[y == 0].mean() < 0.05


# ---------------------------------------------------------------- features


def test_feature_shapes_and_layout(table):
    X = build_features(table)
    assert X.shape == (len(table), 20) and X.dtype == np.float32
    Xe = build_features(table, engineered=True)
    assert Xe.shape == (len(table), 24)
    assert feature_names() == [f"kp_{k}_x" for k in range(10)] + [f"kp_{k}_y" for k in range(10)]
    assert len(feature_names(True)) == 24
    single = normalised_features(seated())
    assert single.shape == (20,)
    assert single[KP["head"]] == pytest.approx(round(300 / 1280, 3))
    assert single[10 + KP["head"]] == pytest.approx(round(300 / 720, 3))  # y / 720, not 780


def test_missing_keypoints_encoding():
    p = seated()
    p[KP["left_hand"]] = (0, 0)
    f = normalised_features(p)
    assert f[KP["left_hand"]] == MISSING and f[10 + KP["left_hand"]] == MISSING
    assert normalised_features(p, missing_value=0.0)[KP["left_hand"]] == 0.0


def test_generalise_shifts_by_roi_left(table):
    p = seated(cx=500)
    g = generalised_features(p, roi_left=370)
    assert g[KP["head"]] == pytest.approx(round(130 / 1280, 3))
    assert g[10 + KP["head"]] == pytest.approx(round(300 / 720, 3))
    layout = CameraLayout.from_yaml(CAMERA)
    left = roi_left_from_layout(table, layout)
    assert set(np.unique(left)) == {d.roi[0] for d in layout.desks}
    X = build_features(table, "generalise", roi_left=left)
    assert X.shape == (len(table), 20)
    with pytest.raises(ValueError):
        build_features(table, "generalise")
    with pytest.raises(ValueError):
        build_features(table, "bogus")


def test_engineered_features():
    e = engineered_features(raised("left"))
    assert e[0] > 130 / 180 and e[2] == 1.0  # left arm straight and above shoulder
    assert e[3] == 0.0
    p = seated()
    p[KP["right_elbow"]] = (0, 0)
    e = engineered_features(p)
    assert e[1] == MISSING and e[3] == 0.0


# ---------------------------------------------------------------- data validation


def test_trailing_space_header_is_repaired():
    a = pd.DataFrame({"frame": [0, 1], "hand_raised": [0, 1]})
    b = pd.DataFrame({"frame": [2, 3], "hand_raised ": [1, 0]})
    naive = pd.concat([a, b], ignore_index=True)
    assert naive["hand_raised"].isna().sum() == 2  # the original bug: phantom NaN labels
    with pytest.raises(ValueError, match="trailing whitespace"):
        validate_binary_labels(naive["hand_raised"])
    fixed = clean_columns(naive)
    assert list(fixed.columns) == ["frame", "hand_raised"]
    assert validate_binary_labels(fixed["hand_raised"]).tolist() == [0, 1, 1, 0]
    clash = pd.DataFrame([[0, 1]], columns=["hand_raised", "hand_raised "])
    with pytest.raises(ValueError, match="disagree"):
        clean_columns(clash)


def test_label_validation():
    assert validate_binary_labels(["0", " 1", 1.0, True]).tolist() == [0, 1, 1, 1]
    with pytest.raises(ValueError, match="binary"):
        validate_binary_labels([0, 1, 2])
    with pytest.raises(ValueError, match="binary"):
        validate_binary_labels(["yes", "0"])


def test_load_tables_per_person_format(tmp_path):
    rows = []
    for f in range(4):
        row = {"frame": f, "person": "person1", "hand_raised ": f % 2}
        row.update({n: f"({10 + k},{20 + k})" for k, n in enumerate(KEYPOINT_NAMES)})
        rows.append(row)
    path = tmp_path / "vid_a.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    path_b = tmp_path / "vid_b.csv"
    pd.DataFrame(rows).rename(columns={"hand_raised ": "hand_raised"}).to_csv(path_b, index=False)
    df = load_tables([path, path_b], label_column="hand_raised")
    assert df["hand_raised"].tolist() == [0, 1, 0, 1] * 2
    assert set(df["video"]) == {"vid_a", "vid_b"}
    assert df["kp_3_x"].iloc[0] == 13 and df["kp_3_y"].iloc[0] == 23  # both coordinates kept
    parsed = parse_point_columns(pd.DataFrame({n: ["(1,2,0.9)"] for n in KEYPOINT_NAMES}))
    assert parsed["kp_0_x"].iloc[0] == 1 and parsed["kp_0_y"].iloc[0] == 2
    with pytest.raises(ValueError):
        load_tables([])


# ---------------------------------------------------------------- models


def test_mlp_forward_shapes():
    x = torch.randn(7, 20)
    assert KeypointMLP()(x).shape == (7, 1)
    assert KeypointMLP(input_dim=24)(torch.randn(3, 24)).shape == (3, 1)
    tb = TwoBranchMLP().eval()
    out = tb(x)
    assert out.shape == (7, 1)
    assert torch.allclose(out, tb(x[:, :10], x[:, 10:]))  # split == explicit two inputs
    with pytest.raises(ValueError):
        tb(torch.randn(2, 24))


def test_multilabel_accuracy_bounded():
    logits = torch.tensor([[5.0, -5.0, 5.0, -5.0], [5.0, 5.0, -5.0, -5.0]])
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    assert multilabel_accuracy(logits, labels) == 1.0  # original formula would give 4.0
    assert multilabel_accuracy(logits, 1 - labels) == 0.0
    with pytest.raises(ValueError):
        multilabel_accuracy(logits, labels[:, :2])


def test_random_forest_and_grid_search(table):
    X = build_features(table)
    y = table["hand_raise"].to_numpy()
    rf = make_random_forest()
    assert rf.n_estimators == 10 and rf.random_state == 42
    assert len(RF_PARAM_GRID["n_estimators"]) == 3
    grid = {"n_estimators": [5, 10], "max_depth": [None, 5], "random_state": [0]}
    search = grid_search_random_forest(X, y, table["video"], grid, scoring="recall", n_splits=3)
    scores = search.cv_results_["mean_test_score"]
    assert np.isfinite(scores).all()  # the original search returned NaN for every candidate
    assert search.best_score_ > 0
    with pytest.raises(ValueError):
        grid_search_random_forest(X, np.where(y == 1, 2, 0), table["video"], grid, n_splits=3)
    with pytest.raises(ValueError):
        grid_search_random_forest(X, y, table["video"], grid, scoring="accuracy")


def test_hybrid_model_forward():
    pytest.importorskip("torchvision")
    from qiyas_akd.hand_raise import HybridModel

    model = HybridModel().eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 224, 224), torch.randn(2, 20)).shape == (2, 1)
    four = HybridModel(keypoint_dim=80, num_outputs=4).eval()
    with torch.no_grad():
        assert four(torch.randn(1, 3, 224, 224), torch.randn(1, 80)).shape == (1, 4)


# ---------------------------------------------------------------- training


@pytest.mark.parametrize("model", ["two_branch", "mlp", "rf", "rule"])
def test_one_epoch_synthetic_training(table, model):
    cfg = TrainConfig(model=model, epochs=1, seed=3, test_size=0.34)
    res = train(table, cfg)
    m = res.metrics
    assert m["n_train"] + m["n_val"] == len(table)
    train_videos = set(table["video"]) - set(m["val_groups"])
    assert train_videos and set(m["val_groups"])  # grouped split: both sides non-empty
    assert {"frame", "event"} <= set(m) and 0.0 <= m["frame"]["f1"] <= 1.0
    if model == "rf":
        assert m["frame"]["recall"] > 0.5
    if model in ("two_branch", "mlp"):
        assert m["epochs_run"] == 1 and m["threshold"] == 0.6


def test_training_is_deterministic(table):
    cfg = TrainConfig(model="mlp", epochs=2, seed=5)
    a, b = train(table, cfg).metrics, train(table, cfg).metrics
    assert a["train_loss"] == b["train_loss"] and a["frame"] == b["frame"]


def test_generalise_training_needs_layout(table):
    cfg = TrainConfig(model="rf", variant="generalise", engineered=True)
    with pytest.raises(ValueError, match="layout"):
        train(table, cfg)
    res = train(table, cfg, CameraLayout.from_yaml(CAMERA))
    assert res.metrics["frame"]["tp"] > 0


def test_config_validation_and_yaml(tmp_path):
    with pytest.raises(ValueError):
        TrainConfig(model="svm")
    with pytest.raises(ValueError):
        TrainConfig(model="two_branch", engineered=True)
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"modle": "rf"})
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("model: rf\nn_estimators: 7\n", encoding="utf8")
    from qiyas_akd.hand_raise.train import _parser

    args = _parser().parse_args(["--synthetic", "--config", str(cfg_file), "--n-estimators", "9"])
    cfg = resolve_config(args)
    assert cfg.model == "rf" and cfg.n_estimators == 9  # CLI overrides YAML
    for name in ("two_branch.yaml", "rf_generalise.yaml"):
        path = ROOT / "configs" / "hand_raise" / name
        args = _parser().parse_args(["--synthetic", "--config", str(path)])
        assert isinstance(resolve_config(args), TrainConfig)


def test_cli_synthetic_and_save(tmp_path, capsys):
    out = tmp_path / "mlp.pt"
    rc = main(["--synthetic", "--model", "mlp", "--epochs", "1", "--n-videos", "4",
               "--n-frames", "60", "--save", str(out)])
    assert rc == 0 and out.exists()
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["model"] == "mlp" and "event" in metrics
    state = torch.load(out, weights_only=False)
    assert state["config"]["model"] == "mlp"
    rf_out = tmp_path / "rf.joblib"
    rc = main(["--synthetic", "--model", "rf", "--n-videos", "4", "--n-frames", "60",
               "--save", str(rf_out)])
    assert rc == 0 and rf_out.exists()
    capsys.readouterr()
    rule = train(generate_frame_table("hand_raise", 3, 2, 40, seed=1), TrainConfig(model="rule"))
    with pytest.raises(ValueError):
        save_model(rule, tmp_path / "rule.pt")
