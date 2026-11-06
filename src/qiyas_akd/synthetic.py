"""Synthetic seated-candidate keypoint sequences.

The real dataset (exam-hall footage of real candidates) cannot be published, so every pipeline in
this repo can run end-to-end on synthetic data with the same schema. The generator is a simple
kinematic model — it is for exercising code and tests, **not** for reproducing reported results.

Behaviours modelled (per desk, per frame):

* ``looking_around`` — head yaw: the ear line tilts beyond the allowed angle;
* ``hand_raise`` — one hand above the head with the elbow extended;
* ``hand_in_pocket`` — one hand drops below the desk edge / disappears (missing keypoint) at the
  hip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .keypoints import FRAME_H, FRAME_W, KP, NUM_KEYPOINTS

BEHAVIOURS = ("looking_around", "hand_raise", "hand_in_pocket")


def _base_pose(cx: float, cy: float, scale: float) -> np.ndarray:
    """Neutral seated pose (K, 2) in pixels, centred at head position (cx, cy)."""
    s = scale
    p = np.zeros((NUM_KEYPOINTS, 2))
    p[KP["head"]] = (cx, cy)
    p[KP["neck"]] = (cx, cy + 40 * s)
    p[KP["left_ear"]] = (cx - 18 * s, cy + 2 * s)
    p[KP["right_ear"]] = (cx + 18 * s, cy + 2 * s)
    p[KP["left_shoulder"]] = (cx - 45 * s, cy + 55 * s)
    p[KP["right_shoulder"]] = (cx + 45 * s, cy + 55 * s)
    p[KP["left_elbow"]] = (cx - 60 * s, cy + 105 * s)
    p[KP["right_elbow"]] = (cx + 60 * s, cy + 105 * s)
    p[KP["left_hand"]] = (cx - 30 * s, cy + 140 * s)  # hands resting on the desk
    p[KP["right_hand"]] = (cx + 30 * s, cy + 140 * s)
    return p


def _apply(p: np.ndarray, behaviour: str, rng: np.random.Generator, scale: float) -> np.ndarray:
    p = p.copy()
    s = scale
    if behaviour == "looking_around":
        # rotate the ear line by 50-80 degrees around the head point
        ang = np.deg2rad(rng.uniform(50, 80) * rng.choice([-1, 1]))
        c, sn = np.cos(ang), np.sin(ang)
        for k in ("left_ear", "right_ear"):
            v = p[KP[k]] - p[KP["head"]]
            p[KP[k]] = p[KP["head"]] + (c * v[0] - sn * v[1], sn * v[0] + c * v[1])
    elif behaviour == "hand_raise":
        side = rng.choice(["left", "right"])
        sh = p[KP[f"{side}_shoulder"]]
        p[KP[f"{side}_elbow"]] = sh + (rng.normal(0, 5), -55 * s)
        p[KP[f"{side}_hand"]] = sh + (rng.normal(0, 5), -115 * s)
    elif behaviour == "hand_in_pocket":
        side = rng.choice(["left", "right"])
        sh = p[KP[f"{side}_shoulder"]]
        sign = -1 if side == "left" else 1
        p[KP[f"{side}_elbow"]] = sh + (sign * 30 * s, 60 * s)
        if rng.random() < 0.5:
            p[KP[f"{side}_hand"]] = (0.0, 0.0)  # hand occluded below the desk → missing
        else:
            p[KP[f"{side}_hand"]] = sh + (sign * 10 * s, 120 * s)
    else:
        raise ValueError(f"unknown behaviour {behaviour!r}")
    return p


def generate_desk_sequence(
    n_frames: int,
    behaviour: str,
    rng: np.random.Generator,
    desk_x: float = 320.0,
    event_rate: float = 0.02,
    min_event: int = 6,
    max_event: int = 20,
    noise_px: float = 2.0,
    drop_prob: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """One desk's keypoint track: returns ``(xy (T, K, 2) pixels, labels (T,) 0/1)``."""
    if behaviour not in BEHAVIOURS:
        raise ValueError(f"unknown behaviour {behaviour!r}")
    scale = rng.uniform(0.8, 1.2)
    base = _base_pose(desk_x + rng.normal(0, 10), FRAME_H * 0.45 + rng.normal(0, 10), scale)
    xy = np.zeros((n_frames, NUM_KEYPOINTS, 2))
    labels = np.zeros(n_frames, dtype=int)
    t = 0
    while t < n_frames:
        if rng.random() < event_rate:
            length = int(rng.integers(min_event, max_event + 1))
            labels[t : t + length] = 1
            t += length
        else:
            t += 1
    for i in range(n_frames):
        pose = _apply(base, behaviour, rng, scale) if labels[i] else base
        missing = (pose == 0).all(axis=-1)
        pose = pose + rng.normal(0, noise_px, pose.shape)
        pose[missing] = 0.0
        pose[rng.random(NUM_KEYPOINTS) < drop_prob] = 0.0
        xy[i] = np.clip(pose, 0, [FRAME_W - 1, FRAME_H - 1])
    return xy, labels


def generate_frame_table(
    behaviour: str,
    n_videos: int = 6,
    n_desks: int = 4,
    n_frames: int = 120,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-frame, per-desk table in the same long format as the real annotation CSVs.

    Columns: ``video, frame, desk_no, position_a..d, kp_{k}_x, kp_{k}_y (pixels), <behaviour>``.
    ``position_*`` is a one-hot-ish descriptor of where the desk sits in the camera view.
    """
    rng = np.random.default_rng(seed)
    rows = []
    desk_xs = np.linspace(160, FRAME_W - 160, n_desks)
    for v in range(n_videos):
        for d in range(n_desks):
            xy, lab = generate_desk_sequence(n_frames, behaviour, rng, desk_x=desk_xs[d])
            pos = np.zeros(4)
            pos[d % 4] = 1.0
            for f in range(n_frames):
                row = {"video": f"v{v}", "frame": f, "desk_no": d + 1}
                row.update({f"position_{c}": pos[i] for i, c in enumerate("abcd")})
                for k in range(NUM_KEYPOINTS):
                    row[f"kp_{k}_x"] = xy[f, k, 0]
                    row[f"kp_{k}_y"] = xy[f, k, 1]
                row[behaviour] = int(lab[f])
                rows.append(row)
    return pd.DataFrame(rows)
