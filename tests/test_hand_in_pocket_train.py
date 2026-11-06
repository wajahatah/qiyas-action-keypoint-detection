"""Hand in pocket — training (1 epoch on synthetic data), grouped split, streaming inference,
CLI."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from qiyas_akd.evaluation import grouped_split
from qiyas_akd.hand_in_pocket.dataprep.pipeline import (
    DataConfig,
    build_windows,
    finalise,
    prepare_frames,
)
from qiyas_akd.hand_in_pocket.dataprep.windowing import WindowSpec
from qiyas_akd.hand_in_pocket.infer import (
    KeypointPredictor,
    StreamingPredictor,
    evaluate_events,
    frame_events,
    load_predictor,
    predict_video_table,
)
from qiyas_akd.hand_in_pocket.models import GRUClassifier, WindowMLP, make_tree_model
from qiyas_akd.hand_in_pocket.train import (
    ImageConfig,
    SyntheticConfig,
    TrainConfig,
    main,
    run,
    synthetic_layout,
)
from qiyas_akd.keypoints import NUM_KEYPOINTS
from qiyas_akd.synthetic import generate_frame_table

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "hand_in_pocket"
LABEL = "hand_in_pocket"
QUIET = {"log": lambda _msg: None}


def small(name: str, **model_overrides) -> TrainConfig:
    cfg = TrainConfig.from_yaml(CONFIGS / f"{name}.yaml")
    cfg = replace(
        cfg,
        synthetic=SyntheticConfig(n_videos=4, n_desks=2, n_frames=80, seed=0),
        train=replace(cfg.train, epochs=1),
    )
    if model_overrides:
        cfg = replace(cfg, model=replace(cfg.model, **model_overrides))
    return cfg


def test_all_configs_parse():
    names = sorted(p.stem for p in CONFIGS.glob("*.yaml"))
    assert names == ["gru", "hybrid_3dcnn", "mlp_5frame", "mlp_t3", "rf_temporal", "xgboost"]
    t3 = TrainConfig.from_yaml(CONFIGS / "mlp_t3.yaml")
    assert len(t3.data.feature_columns) == 64 and t3.data.windowing.keep == (0, 2, 4)
    assert len(TrainConfig.from_yaml(CONFIGS / "mlp_5frame.yaml").data.feature_columns) == 104


def test_grouped_split_has_no_video_overlap():
    frames = generate_frame_table(LABEL, n_videos=6, n_desks=2, n_frames=40, seed=2)
    w = build_windows(prepare_frames(frames, DataConfig()), DataConfig())
    tr, va = grouped_split(w["video"], 0.3, seed=0)
    assert set(w["video"].iloc[tr]).isdisjoint(set(w["video"].iloc[va]))
    assert len(tr) + len(va) == len(w)


@pytest.mark.parametrize("name", ["mlp_t3", "mlp_5frame", "gru"])
def test_one_epoch_torch_training(tmp_path, name):
    m = run(small(name), synthetic=True, out_dir=tmp_path, **QUIET)
    assert (tmp_path / "model.pt").is_file()
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf8"))
    assert saved["epochs_run"] == 1 and saved["config"]["name"] == name
    assert {"tp", "fp", "fn", "f1"} <= set(m["val_events"]) and m["val_windows"]["n"] > 0
    pred = load_predictor(tmp_path / "model.pt")
    assert len(pred.feature_columns) == m["n_features"]


def test_one_epoch_lstm_with_missing_sentinel(tmp_path):
    cfg = small("gru", kind="lstm", params={"num_layers": 2, "clamp": [-10, 1]}, missing_value=-10)
    m = run(cfg, synthetic=True, out_dir=tmp_path, **QUIET)
    assert m["kind"] == "lstm" and load_predictor(tmp_path / "model.pt").missing_value == -10


def test_random_forest_training_and_grouped_grid_search(tmp_path):
    m = run(
        small("rf_temporal", params={"n_estimators": 10}), synthetic=True, out_dir=tmp_path, **QUIET
    )
    assert (tmp_path / "model.joblib").is_file() and "val_events" in m
    cfg = small("rf_temporal", grid={"n_estimators": [5, 10], "max_depth": [None, 3]}, grid_cv=2)
    run(cfg, synthetic=True, out_dir=tmp_path / "grid", **QUIET)
    assert load_predictor(tmp_path / "grid" / "model.joblib").kind == "rf"


@pytest.mark.parametrize(("module", "kind"), [("xgboost", "xgboost"), ("lightgbm", "lightgbm")])
def test_boosting_training(tmp_path, module, kind):
    pytest.importorskip(module)
    cfg = small("xgboost", kind=kind, params={"n_estimators": 10})
    m = run(cfg, synthetic=True, out_dir=tmp_path, **QUIET)
    assert m["kind"] == kind and m["n_features"] == 100


@pytest.mark.parametrize("kind", ["hybrid3d", "hybrid2d", "temporal_crop_cnn3d", "temporal_cnn"])
def test_one_epoch_image_models_on_rendered_crops(tmp_path, kind):
    cfg = small("hybrid_3dcnn", kind=kind, params={})
    cfg = replace(
        cfg,
        images=ImageConfig(crop_size=16),
        synthetic=SyntheticConfig(n_videos=3, n_desks=1, n_frames=40, seed=1),
    )
    m = run(cfg, synthetic=True, out_dir=tmp_path, **QUIET)
    assert (tmp_path / "model.pt").is_file() and "val_events" not in m


def _predictor(kind: str, data: DataConfig) -> KeypointPredictor:
    torch.manual_seed(0)
    n = len(data.feature_columns)
    if kind == "mlp":
        return KeypointPredictor(WindowMLP(n, "c2"), "mlp", data)
    if kind == "gru":
        return KeypointPredictor(GRUClassifier(24), "gru", data, missing_value=-10)
    rf = make_tree_model("rf", {"n_estimators": 5}, seed=0)
    rng = np.random.default_rng(0)
    rf.fit(rng.random((40, n)), np.r_[np.zeros(20), np.ones(20)])
    return KeypointPredictor(rf, "rf", data)


@pytest.mark.parametrize(
    ("kind", "spec", "normalise"),
    [
        ("mlp", WindowSpec(window=5, keep=(0, 2, 4)), "frame"),
        ("mlp", WindowSpec(window=4, layout="time_major"), "roi"),
        ("gru", WindowSpec(window=3), "frame"),
        ("rf", WindowSpec(window=5, keep=(0, 4)), "none"),
    ],
)
def test_streaming_inference_equals_batch_inference(kind, spec, normalise):
    frames = generate_frame_table(LABEL, n_videos=2, n_desks=3, n_frames=30, seed=4)
    layout = synthetic_layout(3)
    data = DataConfig(windowing=spec, normalise=normalise)
    pred = _predictor(kind, data)
    prepared = prepare_frames(frames, data)
    windows = finalise(build_windows(prepared, data), data, layout)
    batch = windows[["video", "desk_no", "frame_end"]].assign(batch=pred.predict_windows(windows))
    stream = predict_video_table(prepared, pred, layout)
    merged = batch.merge(
        stream, left_on=["video", "desk_no", "frame_end"], right_on=["video", "desk_no", "frame"]
    )
    assert len(merged) == len(batch) == stream["prob"].notna().sum()
    np.testing.assert_allclose(merged["batch"], merged["prob"], rtol=0, atol=1e-6)
    first = stream.groupby(["video", "desk_no"]).head(spec.window - 1)
    assert first["prob"].isna().all()


def test_streaming_missing_detection_policies():
    data = DataConfig(windowing=WindowSpec(window=3))
    pred = _predictor("mlp", data)
    xy = np.full((NUM_KEYPOINTS, 2), 300.0)
    pos = {1: [1, 0, 0, 0]}
    for policy, expect_after_gap in (("reset", set()), ("keep", {1})):
        s = StreamingPredictor(pred, on_missing=policy)
        assert s.update({1: xy}, pos) == {} and s.update({1: xy}, pos) == {}
        assert set(s.update({1: xy}, pos)) == {1}  # buffer full → probability
        s.update({}, pos)  # desk missing in this frame
        assert set(s.update({1: xy}, pos)) == expect_after_gap
    s = StreamingPredictor(pred)
    for _ in range(3):
        out = s.update({1: xy, 2: np.zeros((NUM_KEYPOINTS, 2))}, {1: [0] * 4, 2: [0] * 4})
    assert set(out) == {1}  # all-missing person counts as no detection
    with pytest.raises(ValueError):
        StreamingPredictor(_predictor("mlp", DataConfig(windowing=WindowSpec(random_drop=1))))


def test_event_extraction_and_counts():
    pred = pd.DataFrame(
        {
            "video": "v",
            "desk_no": 1,
            "frame": range(10),
            "pred": [0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
            LABEL: [0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
        }
    )
    assert frame_events(pred)[("v", 1)] == [(1, 3), (6, 6)]
    assert frame_events(pred, min_len=2)[("v", 1)] == [(1, 3)]
    c = evaluate_events(pred, LABEL)
    assert (c.tp, c.fp, c.fn) == (1, 1, 1)


def test_cli_synthetic_end_to_end(tmp_path):
    metrics = main(
        [
            "--config",
            str(CONFIGS / "mlp_t3.yaml"),
            "--synthetic",
            "--epochs",
            "1",
            "--out",
            str(tmp_path / "inproc"),
        ]
    )
    assert metrics["n_features"] == 64
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = tmp_path / "cli"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "qiyas_akd.hand_in_pocket.train",
            "--config",
            str(CONFIGS / "mlp_t3.yaml"),
            "--synthetic",
            "--epochs",
            "1",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "model.pt").is_file() and (out / "metrics.json").is_file()
    assert "val_events" in proc.stdout
