"""Throughput benchmark: N parallel model instances (processes) x batch size B.

Why this exists: the deployment question was "how many camera streams can one GPU serve?", which
depends on both the batch size an engine is built for and on how many engine instances run side by
side. The original measurement ran 1-2 copies of a script, each looping over a video, feeding
``[frame] * B`` to a static batch-B TensorRT engine and printing ``1 / (loop time)`` as "fps".

That number was ambiguous in two ways, both fixed here:

1. **batches/s, not frames/s** — one loop iteration processed B frames, so the printed "fps" was
   batches per second; frames per second is B times larger. This benchmark reports both,
   ``frames_per_s == batches_per_s * batch_size`` exactly.
2. **not inference-only** — the timed span included keypoint drawing, the looking-around geometry,
   ``cv2.resize``/``imshow`` and ``waitKey``. Here only the model call is timed (per-batch latency
   statistics), and throughput is measured over a synchronised window in which the workers do
   nothing but call the model.

Each worker is a separate *process* (``spawn`` start method, safe with CUDA) that builds its own
model from a picklable ``model_factory`` — like separate deployment workers each owning an engine.
Any callable ``model(list_of_frames)`` works, so the benchmark is testable with
:func:`dummy_model` and no GPU.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

import numpy as np

from ..keypoints import FRAME_H, FRAME_W
from ._optional import import_yolo

Model = Callable[[list[np.ndarray]], Any]
ModelFactory = Callable[[], Model]


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregate over all workers. Latencies are per *batch* and inference-only."""

    n_workers: int
    batch_size: int
    n_batches_per_worker: int
    wall_time_s: float
    batches_per_s: float
    frames_per_s: float
    frames_per_s_per_worker: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float


def _make_batch(batch_size: int, frame_shape: tuple[int, int, int], seed: int) -> list[np.ndarray]:
    """One deterministic random frame repeated ``batch_size`` times (as in the original runs)."""
    frame = np.random.default_rng(seed).integers(0, 256, size=frame_shape, dtype=np.uint8)
    return [frame] * batch_size


def _run_worker(
    model_factory: ModelFactory,
    batch_size: int,
    n_batches: int,
    warmup: int,
    frame_shape: tuple[int, int, int],
    seed: int,
    barrier: Any | None,
) -> tuple[list[float], float, float]:
    """Build the model, warm up, wait for all workers, then time ``n_batches`` model calls.

    Returns ``(per-batch latencies in s, window start, window end)``; window times are
    ``time.time()`` so they are comparable across processes (its resolution can be coarse on some
    platforms, so use enough batches for a window of at least a few seconds).
    """
    model = model_factory()
    batch = _make_batch(batch_size, frame_shape, seed)
    for _ in range(warmup):  # first calls include CUDA context / engine allocation
        model(batch)
    if barrier is not None:
        barrier.wait()
    latencies = []
    start = time.time()
    for _ in range(n_batches):
        t0 = time.perf_counter()
        model(batch)
        latencies.append(time.perf_counter() - t0)
    return latencies, start, time.time()


def _process_entry(queue: Any, barrier: Any, *args: Any) -> None:
    try:
        queue.put(("ok", _run_worker(*args, barrier)))
    except Exception as exc:
        # Break the barrier so the other workers stop waiting, and report instead of hanging.
        barrier.abort()
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def run_benchmark(
    model_factory: ModelFactory,
    batch_size: int = 1,
    n_workers: int = 1,
    n_batches: int = 100,
    warmup: int = 5,
    frame_shape: tuple[int, int, int] = (FRAME_H, FRAME_W, 3),
    seed: int = 0,
    timeout_s: float = 600.0,
) -> BenchmarkResult:
    """Run ``n_workers`` model instances in parallel, each timing ``n_batches`` batches.

    ``n_workers == 1`` runs in the current process (no process start-up cost); more workers run in
    spawned processes, so ``model_factory`` must be picklable (a module-level function, a
    ``functools.partial`` of one, or an instance of a module-level class).
    """
    if batch_size < 1 or n_workers < 1 or n_batches < 1 or warmup < 0:
        raise ValueError("batch_size, n_workers, n_batches >= 1 and warmup >= 0 are required")
    if n_workers == 1:
        args = (model_factory, batch_size, n_batches, warmup, frame_shape, seed, None)
        outputs = [_run_worker(*args)]
    else:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        barrier = ctx.Barrier(n_workers)
        procs = [
            ctx.Process(
                target=_process_entry,
                args=(queue, barrier, model_factory, batch_size, n_batches, warmup, frame_shape,
                      seed + i),
            )
            for i in range(n_workers)
        ]
        for p in procs:
            p.start()
        try:
            replies = [queue.get(timeout=timeout_s) for _ in procs]
        finally:
            for p in procs:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
        errors = [msg for status, msg in replies if status == "error"]
        if errors:
            raise RuntimeError(f"benchmark worker failed: {errors[0]}")
        outputs = [payload for _, payload in replies]
    return _aggregate(outputs, batch_size, n_batches)


def _aggregate(
    outputs: Sequence[tuple[list[float], float, float]], batch_size: int, n_batches: int
) -> BenchmarkResult:
    latencies = np.concatenate([np.asarray(lat) for lat, _, _ in outputs])
    window = max(end for _, _, end in outputs) - min(start for _, start, _ in outputs)
    window = max(window, float(latencies.max()), 1e-9)  # guard against clock resolution
    total_batches = n_batches * len(outputs)
    batches_per_s = total_batches / window
    per_worker = float(np.mean([batch_size * len(lat) / sum(lat) for lat, _, _ in outputs]))
    return BenchmarkResult(
        n_workers=len(outputs),
        batch_size=batch_size,
        n_batches_per_worker=n_batches,
        wall_time_s=window,
        batches_per_s=batches_per_s,
        frames_per_s=batches_per_s * batch_size,
        frames_per_s_per_worker=per_worker,
        latency_ms_mean=float(latencies.mean() * 1e3),
        latency_ms_p50=float(np.percentile(latencies, 50) * 1e3),
        latency_ms_p95=float(np.percentile(latencies, 95) * 1e3),
    )


# --------------------------------------------------------------------------------------------------
# Model factories (module level so they pickle into spawned workers)
# --------------------------------------------------------------------------------------------------


def _dummy_call(batch: list[np.ndarray], latency_s: float) -> list[None]:
    time.sleep(latency_s)
    return [None] * len(batch)


def dummy_model(latency_s: float = 0.002) -> Model:
    """Stand-in model whose call takes ``latency_s`` regardless of batch size (for tests/CLI)."""
    return partial(_dummy_call, latency_s=latency_s)


@dataclass(frozen=True)
class YoloFactory:
    """Picklable factory that loads a YOLO-pose ``.pt`` / ``.engine`` inside each worker."""

    weights: str
    device: int | str | None = None
    half: bool = False

    def __call__(self) -> Model:
        model = import_yolo()(self.weights, task="pose")
        kwargs: dict[str, Any] = {"verbose": False, "half": self.half}
        if self.device is not None:
            kwargs["device"] = self.device
        return partial(model, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m qiyas_akd.pose.benchmark --weights best_fp16_b10.engine --batch 10``."""
    ap = argparse.ArgumentParser(description="Pose-model throughput benchmark")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights", help=".pt checkpoint or TensorRT .engine")
    src.add_argument("--dummy-latency-ms", type=float, help="benchmark a sleeping dummy model")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--half", action="store_true", help="FP16 for .pt models (engines are fixed)")
    a = ap.parse_args(argv)
    if a.weights:
        factory: ModelFactory = YoloFactory(a.weights, a.device, a.half)
    else:
        factory = partial(dummy_model, a.dummy_latency_ms / 1e3)
    res = run_benchmark(factory, a.batch, a.workers, a.batches, a.warmup)
    print(json.dumps(asdict(res), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
