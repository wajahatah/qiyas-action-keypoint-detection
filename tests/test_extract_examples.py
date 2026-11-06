import importlib.util
from pathlib import Path

import pandas as pd

from qiyas_akd.extract import frame_table_columns
from qiyas_akd.synthetic import BEHAVIOURS

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "make_synthetic_dataset.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("make_synthetic_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_synthetic_dataset_writes_all_behaviours(tmp_path):
    out = tmp_path / "synthetic"
    args = ["--out", str(out), "--videos", "2", "--desks", "2", "--frames", "10"]
    assert _load_example().main(args) == 0
    for behaviour in BEHAVIOURS:
        df = pd.read_csv(out / f"{behaviour}.csv")
        assert len(df) == 2 * 2 * 10
        assert list(df.columns) == frame_table_columns() + [behaviour]
        assert set(df[behaviour].unique()) <= {0, 1}


def test_make_synthetic_dataset_is_deterministic(tmp_path):
    example = _load_example()
    for d in ("a", "b"):
        example.main(["--out", str(tmp_path / d), "--videos", "1", "--frames", "5",
                      "--behaviours", "hand_in_pocket"])
    a = (tmp_path / "a" / "hand_in_pocket.csv").read_text()
    assert a == (tmp_path / "b" / "hand_in_pocket.csv").read_text()
    assert not (tmp_path / "a" / "hand_raise.csv").exists()
