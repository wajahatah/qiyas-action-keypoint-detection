from functools import partial

import pytest

from qiyas_akd.pose.benchmark import YoloFactory, _aggregate, dummy_model, run_benchmark

SMALL_FRAME = (36, 64, 3)


def test_single_worker_dummy_model():
    res = run_benchmark(
        partial(dummy_model, 0.002), batch_size=5, n_workers=1, n_batches=10, warmup=1,
        frame_shape=SMALL_FRAME,
    )
    assert res.n_workers == 1 and res.batch_size == 5
    assert res.frames_per_s > 0 and res.batches_per_s > 0
    assert res.frames_per_s == pytest.approx(res.batches_per_s * 5)
    assert res.latency_ms_mean >= 2.0  # inference-only latency includes the model's sleep
    assert res.latency_ms_p95 >= res.latency_ms_p50 > 0
    assert res.frames_per_s_per_worker > 0


def test_parallel_workers_in_processes():
    res = run_benchmark(
        partial(dummy_model, 0.005), batch_size=4, n_workers=2, n_batches=8, warmup=1,
        frame_shape=SMALL_FRAME, timeout_s=120,
    )
    assert res.n_workers == 2
    assert res.frames_per_s == pytest.approx(res.batches_per_s * 4)
    assert res.frames_per_s > 0


def test_worker_failure_is_reported(tmp_path):
    # Fails in the child either because Ultralytics is absent or because the engine does not exist;
    # the barrier is aborted so the healthy-looking sibling does not hang the benchmark.
    factory = YoloFactory(str(tmp_path / "missing.engine"))
    with pytest.raises(RuntimeError, match="worker failed"):
        run_benchmark(factory, n_workers=2, n_batches=2, frame_shape=SMALL_FRAME, timeout_s=120)


def test_aggregate_math():
    # two workers, 4 batches of 0.1 s each, fully overlapping 0.4 s window, batch 10
    outs = [([0.1] * 4, 100.0, 100.4), ([0.1] * 4, 100.0, 100.4)]
    res = _aggregate(outs, batch_size=10, n_batches=4)
    assert res.batches_per_s == pytest.approx(8 / 0.4)
    assert res.frames_per_s == pytest.approx(10 * 8 / 0.4)
    assert res.frames_per_s_per_worker == pytest.approx(100.0)
    assert res.latency_ms_mean == pytest.approx(100.0)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        run_benchmark(partial(dummy_model, 0.0), batch_size=0)
