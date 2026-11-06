import sys
from pathlib import Path

import pytest

from qiyas_akd.pose import export as export_mod
from qiyas_akd.pose import train as train_mod
from qiyas_akd.pose.benchmark import YoloFactory
from qiyas_akd.pose.infer import UltralyticsPoseModel

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "pose"


@pytest.fixture
def no_ultralytics(monkeypatch):
    """Make ``import ultralytics`` fail even if it happens to be installed."""
    monkeypatch.setitem(sys.modules, "ultralytics", None)


def test_train_config_file_maps_to_ultralytics_kwargs():
    cfg = train_mod.load_train_config(CONFIGS / "yolo11l_pose.yaml", data="d/data.yaml")
    kw = train_mod.to_ultralytics_kwargs(cfg)
    assert cfg.model == "yolo11l-pose.pt"
    assert kw["data"] == "d/data.yaml"
    assert kw["imgsz"] == 640 and kw["epochs"] == 300 and kw["optimizer"] == "auto"
    assert kw["batch"] == 0.95 and kw["patience"] == 5 and kw["lr0"] == 0.001
    assert kw["seed"] == 0 and kw["deterministic"] is True
    assert "model" not in kw and "extra" not in kw
    assert "freeze" not in kw and "device" not in kw  # None values are dropped


def test_train_config_validation(tmp_path):
    with pytest.raises(ValueError, match="data"):
        train_mod.to_ultralytics_kwargs(train_mod.TrainConfig(model="m.pt"))
    with pytest.raises(ValueError):
        train_mod.TrainConfig(model="m.pt", batch=1.5)
    with pytest.raises(ValueError, match="extra repeats"):
        train_mod.TrainConfig(model="m.pt", extra={"epochs": 3})
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: m.pt\nlearning_rate: 1\n", encoding="utf8")
    with pytest.raises(ValueError, match="unknown keys"):
        train_mod.load_train_config(bad)
    cfg = train_mod.TrainConfig(model="m.pt", data="d.yaml", extra={"fliplr": 0.5})
    assert train_mod.to_ultralytics_kwargs(cfg)["fliplr"] == 0.5


def test_train_without_ultralytics_raises_clear_error(no_ultralytics):
    cfg = train_mod.TrainConfig(model="m.pt", data="d.yaml")
    with pytest.raises(ImportError, match=r"qiyas-akd\[pose\]"):
        train_mod.train(cfg)


def test_export_config_and_names():
    cfg = export_mod.load_export_config(CONFIGS / "export_tensorrt_fp16.yaml", weights="w/best.pt")
    kw = export_mod.export_kwargs(cfg)
    assert kw == {
        "format": "engine",
        "half": True,
        "batch": 10,
        "dynamic": False,
        "imgsz": 640,
        "opset": 19,
        "device": 0,
    }
    assert export_mod.artefact_name(cfg) == "best_fp16_b10.engine"
    dyn = export_mod.ExportConfig(
        weights="best.pt", format="onnx", half=False, batch=5, dynamic=True
    )
    assert export_mod.artefact_name(dyn) == "best_fp32_b5_dyn.onnx"
    with pytest.raises(ValueError):
        export_mod.ExportConfig(weights="best.pt", format="tflite")
    with pytest.raises(ValueError, match="weights"):
        export_mod.load_export_config(CONFIGS / "export_tensorrt_fp16.yaml")


def test_export_without_ultralytics_raises_clear_error(no_ultralytics):
    with pytest.raises(ImportError, match=r"qiyas-akd\[pose\]"):
        export_mod.export(export_mod.ExportConfig(weights="best.pt"))


def test_inference_wrappers_without_ultralytics_raise_clear_error(no_ultralytics):
    with pytest.raises(ImportError, match="Ultralytics"):
        UltralyticsPoseModel("best.pt")
    with pytest.raises(ImportError, match="Ultralytics"):
        YoloFactory("best.engine")()


def test_train_and_export_call_ultralytics(monkeypatch, tmp_path):
    """With a fake Ultralytics, the wrappers pass the config through and rename the engine."""
    calls = {}

    class FakeYOLO:
        def __init__(self, weights, task=None):
            calls["init"] = (weights, task)

        def train(self, **kw):
            calls["train"] = kw
            return "trained"

        def export(self, **kw):
            calls["export"] = kw
            out = tmp_path / "best.engine"
            out.write_bytes(b"engine")
            return str(out)

    monkeypatch.setattr(train_mod, "import_yolo", lambda: FakeYOLO)
    monkeypatch.setattr(export_mod, "import_yolo", lambda: FakeYOLO)
    assert train_mod.train(train_mod.TrainConfig(model="m.pt", data="d.yaml")) == "trained"
    assert calls["train"]["data"] == "d.yaml"
    path = export_mod.export(export_mod.ExportConfig(weights=str(tmp_path / "best.pt")))
    assert path.name == "best_fp16_b10.engine" and path.exists()
    assert calls["init"] == (str(tmp_path / "best.pt"), "pose")
    assert calls["export"]["format"] == "engine" and calls["export"]["half"] is True
