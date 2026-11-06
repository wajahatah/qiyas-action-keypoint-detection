"""Regenerate every figure in docs/figures/ from the curated tables in results/.

Figures are derived data only — no images of people, no raw data. Precision / recall / F1 are
computed here from TP/FP/FN counts so the CSVs hold a single source of truth.

Usage:
    python scripts/make_figures.py            # write docs/figures/*.png
    python scripts/make_figures.py --check    # render to a temp dir; fail if any figure errors
                                              # or if docs/figures/ is missing an expected file
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"

# Reference palette (validated: first three categorical slots pass all-pairs CVD checks).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#a3a29c"  # context marks (baselines, superseded approaches)
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "font.size": 10,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def prf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["precision"] = df.tp / (df.tp + df.fp).replace(0, np.nan)
    if "fn" in df:
        df["recall"] = df.tp / (df.tp + df.fn).replace(0, np.nan)
        df["f1"] = 2 * df.precision * df.recall / (df.precision + df.recall)
    return df


def _finish(fig: plt.Figure, out: Path, note: str | None = None) -> Path:
    if note:
        fig.text(0.01, -0.02, note, fontsize=8, color=INK_2, ha="left", va="top")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_hip_bench(out_dir: Path) -> Path:
    """Best variant per model family on the 14-video hand-in-pocket bench (event-level F1)."""
    df = prf(pd.read_csv(RESULTS / "hand_in_pocket_bench14.csv"))
    best = df.sort_values("f1", ascending=False).groupby("family", as_index=False).head(1)
    best = best.sort_values("f1")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    colors = [NEUTRAL if f == "Rule-based baseline" else S1 for f in best.family]
    y = np.arange(len(best))
    ax.barh(y, best.f1, color=colors, height=0.62)
    for yi, (_, r) in zip(y, best.iterrows(), strict=True):
        ax.text(
            r.f1 + 0.01,
            yi,
            f"F1 {r.f1:.2f}  ·  P {r.precision:.2f}  ·  R {r.recall:.2f}",
            va="center",
            fontsize=8.5,
            color=INK_2,
        )
    ax.set_yticks(y, best.family)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Event-level F1 (best variant of each family)")
    ax.set_title("Hand in pocket — best variant of each model family, 14-video test bench")
    return _finish(
        fig,
        out_dir / "hip_bench_f1.png",
        "Alert events counted per video. Grey = rule-based baseline in production at the time. "
        "Full table: results/hand_in_pocket_bench14.csv",
    )


def fig_looking_around(out_dir: Path) -> Path:
    df = prf(pd.read_csv(RESULTS / "looking_around_methods.csv"))
    df = df[df.n_videos == 27].copy()
    df["label"] = np.where(df.pose_model == "-", df.method, df.method + "\n(" + df.pose_model + ")")
    df = df.sort_values("f1")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(df))
    colors = [NEUTRAL if m.startswith("Head-crop") else S1 for m in df.method]
    ax.barh(y, df.f1, color=colors, height=0.62)
    for yi, (_, r) in zip(y, df.iterrows(), strict=True):
        ax.text(
            r.f1 + 0.01,
            yi,
            f"F1 {r.f1:.2f}  ·  P {r.precision:.2f}  ·  R {r.recall:.2f}",
            va="center",
            fontsize=8.5,
            color=INK_2,
        )
    ax.set_yticks(y, df.label, fontsize=8.5)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Event-level F1")
    ax.set_title("Looking around — method comparison on 27 test videos (~8.5 h)")
    return _finish(
        fig,
        out_dir / "looking_around_methods.png",
        "Grey = image-classifier baseline. Ear-line angle on keypoints won and went to production.",
    )


def fig_live_timeline(out_dir: Path) -> Path:
    df = prf(
        pd.read_csv(
            RESULTS / "live_precision_timeline.csv", parse_dates=["period_start", "period_end"]
        )
    )
    df["mid"] = df.period_start + (df.period_end - df.period_start) / 2
    df["n"] = df.tp + df.fp
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for beh, color, name in (
        ("looking_around", S1, "Looking around"),
        ("hand_in_pocket", S2, "Hand in pocket"),
    ):
        d = df[df.behaviour == beh].sort_values("mid")
        gap = d.mid.diff().dt.days.fillna(0).to_numpy() > 90
        seg = np.cumsum(gap)
        for s in np.unique(seg):
            part = d[seg == s]
            ax.plot(part.mid, part.precision, color=color, lw=2, zorder=2)
        for i in np.flatnonzero(gap):
            ax.plot(
                d.mid.iloc[i - 1 : i + 1],
                d.precision.iloc[i - 1 : i + 1],
                color=color,
                lw=1.2,
                ls=(0, (3, 3)),
                zorder=2,
            )
        ax.scatter(
            d.mid,
            d.precision,
            s=36,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=3,
            label=name,
        )
    la = df[df.behaviour == "looking_around"].sort_values("mid")
    ax.annotate(
        f"{la.precision.iloc[-1]:.0%} over {int(la.n.iloc[-1]):,} alerts",
        (la.mid.iloc[-1], la.precision.iloc[-1]),
        xytext=(0, -20),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_2,
        ha="right",
    )
    ax.axvline(pd.Timestamp("2025-09-21"), color=NEUTRAL, lw=1, ls=":", zorder=1)
    ax.text(
        pd.Timestamp("2025-09-21"),
        0.62,
        " keypoint MLP\n replaces rules",
        fontsize=8,
        color=INK_2,
        va="bottom",
    )
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Alert precision (TP / all alerts)")
    ax.set_title("Live system — alert precision over time")
    ax.legend(loc="lower right")
    return _finish(
        fig,
        out_dir / "live_precision_timeline.png",
        "Manual review of every raised alert; missed events were not logged, so recall is unknown. "
        "Dashed = gap with no logs.",
    )


def fig_pose_iterations(out_dir: Path) -> Path:
    df = pd.read_csv(RESULTS / "pose_model_iterations.csv", parse_dates=["train_date"])
    fig, ax = plt.subplots(figsize=(9, 3.9))
    gens = {"G1": (S1, "dataset gen 1"), "G2": (S2, "dataset gen 2"), "G3": (S3, "dataset gen 3")}
    for g, (color, name) in gens.items():
        d = df[df.dataset_generation == g]
        ax.scatter(
            d.train_date,
            d.pose_map50_95,
            s=46,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            label=name,
            zorder=3,
        )
    dep = df[df.deployed_from.notna()]
    nudge = {"v6-1": (-14, 8), "v7-2": (14, 8)}
    for _, r in dep.iterrows():
        ax.annotate(
            r.version,
            (r.train_date, r.pose_map50_95),
            xytext=nudge.get(r.version, (0, 8)),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK_2,
        )
    ax.text(
        pd.Timestamp("2025-04-20"),
        0.9,
        "frame-level split —\nlikely inflated by leakage",
        fontsize=8,
        color=INK_2,
        ha="center",
    )
    ax.set_ylim(0.6, 1.04)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Pose mAP50-95 (validation)")
    ax.set_title("Custom 10-keypoint YOLO-pose — 16 fine-tuning runs")
    ax.legend(loc="lower right")
    return _finish(
        fig,
        out_dir / "pose_iterations.png",
        "Labelled points = versions deployed to the live system. Scores across dataset "
        "generations are not comparable.",
    )


def fig_tensorrt(out_dir: Path) -> Path:
    df = pd.read_csv(RESULTS / "tensorrt_fps.csv")
    df["label"] = (
        df.format.str.replace("PyTorch ", "")
        + " "
        + df.precision
        + "\n"
        + df.parallel_instances.astype(str)
        + "×batch "
        + df.batch.astype(str)
    )
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(df))
    colors = [S1 if p == "FP16" else NEUTRAL for p in df.precision]
    ax.bar(x, df.throughput, color=colors, width=0.6)
    for xi, v in zip(x, df.throughput, strict=True):
        ax.text(xi, v + 0.3, f"{v:g}", ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(x, df.label, fontsize=8)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylabel("Throughput (calls / s, as logged)")
    ax.set_title("TensorRT FP16 export ≈ 2× throughput (YOLO11l-pose, laptop RTX 4060)")
    return _finish(
        fig,
        out_dir / "tensorrt_throughput.png",
        "Logged as 1 / batched-call latency including drawing; the relative speed-up is "
        "the robust claim.",
    )


def fig_hip_live(out_dir: Path) -> Path:
    df = pd.read_csv(RESULTS / "hand_in_pocket_live_reevaluation.csv")
    d = (
        df[df.alert_class == "Hand in pocket"]
        .groupby("approach", as_index=False)[["tp", "fp"]]
        .sum()
    )
    d = prf(d)
    order = [
        "Production rule-based system",
        "MLP + augmentation (this work)",
        "MLP alternative preprocessing",
        "MLP ROI-normalised",
    ]
    d = d.set_index("approach").loc[order].reset_index()
    labels = [
        "Rule-based\n(production)",
        "MLP +\naugmentation",
        "MLP alt.\npreprocessing",
        "MLP ROI-\nnormalised",
    ]
    colors = [NEUTRAL, S1, S1, S1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    x = np.arange(len(d))
    a1.bar(x, d.fp, color=colors, width=0.6)
    for xi, v in zip(x, d.fp, strict=True):
        a1.text(xi, v + 8, f"{v}", ha="center", fontsize=9, color=INK_2)
    a1.set_xticks(x, labels, fontsize=8)
    a1.set_title("False alerts (5 days, 2 rooms)")
    a1.yaxis.grid(True)
    a1.set_axisbelow(True)
    a2.bar(x, d.precision, color=colors, width=0.6)
    for xi, v in zip(x, d.precision, strict=True):
        a2.text(xi, v + 0.01, f"{v:.0%}", ha="center", fontsize=9, color=INK_2)
    a2.set_xticks(x, labels, fontsize=8)
    a2.set_ylim(0, 0.45)
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a2.set_title("Alert precision")
    a2.yaxis.grid(True)
    a2.set_axisbelow(True)
    fig.suptitle(
        "Hand in pocket — live re-evaluation against the rule-based system",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return _finish(
        fig,
        out_dir / "hip_live_reevaluation.png",
        "Same 5 days of live footage replayed through each approach; alerts reviewed "
        "manually (precision only).",
    )


def fig_skeleton(out_dir: Path) -> Path:
    """Schematic of the 10-keypoint skeleton and the ear-line angle (synthetic pose, no photo)."""
    from qiyas_akd.keypoints import KEYPOINT_NAMES, SKELETON
    from qiyas_akd.synthetic import _base_pose

    p = _base_pose(0.0, 0.0, 1.0)
    turned = p.copy()
    ang = np.deg2rad(35)
    c, s = np.cos(ang), np.sin(ang)
    for k in (2, 3):
        v = turned[k] - turned[0]
        turned[k] = turned[0] + (c * v[0] - s * v[1], s * v[0] + c * v[1])
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
    for ax, pose, title in (
        (axes[0], p, "10-keypoint skeleton"),
        (axes[1], turned, "Ear-line angle (head turned)"),
    ):
        for a, b in SKELETON:
            ax.plot(*pose[[a, b]].T, color=NEUTRAL, lw=2, zorder=1)
        ax.scatter(*pose.T, s=40, color=S1, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
    for i, name in enumerate(KEYPOINT_NAMES):
        left = name.startswith("left")
        off = (0, 9) if name == "head" else ((-7, -3) if left else (7, -3))
        ha = "center" if name == "head" else ("right" if left else "left")
        axes[0].annotate(
            name.replace("_", " "),
            p[i],
            xytext=off,
            textcoords="offset points",
            fontsize=7.5,
            color=INK_2,
            ha=ha,
        )
    for ax in axes:
        ax.margins(0.25, 0.12)
    l_ear, r_ear = turned[2], turned[3]
    axes[1].plot(*np.c_[l_ear, r_ear], color=S2, lw=2, zorder=2)
    axes[1].plot([l_ear[0], l_ear[0] + 60], [l_ear[1], l_ear[1]], color=S2, lw=1, ls=":")
    deg = np.degrees(np.arctan2(r_ear[1] - l_ear[1], r_ear[0] - l_ear[0]))
    axes[1].text(
        l_ear[0] + 64, l_ear[1] + 2, f"{abs(deg):.0f}°", color=INK, fontsize=10, va="center"
    )
    return _finish(
        fig,
        out_dir / "skeleton_and_ear_angle.png",
        "Seated, elevated-camera view: legs never visible, so the skeleton spends its "
        "points on head, ears, neck and arms.",
    )


FIGURES_FUNCS = (
    fig_hip_bench,
    fig_looking_around,
    fig_live_timeline,
    fig_pose_iterations,
    fig_tensorrt,
    fig_hip_live,
    fig_skeleton,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            made = [f(Path(tmp)).name for f in FIGURES_FUNCS]
        missing = [m for m in made if not (FIGURES / m).exists()]
        if missing:
            print("docs/figures is missing:", ", ".join(missing))
            return 1
        print(f"{len(made)} figures render; docs/figures is complete.")
        return 0
    FIGURES.mkdir(parents=True, exist_ok=True)
    for f in FIGURES_FUNCS:
        print("wrote", f(FIGURES).relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
