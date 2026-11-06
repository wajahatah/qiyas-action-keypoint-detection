"""Hybrid crop-CNN + keypoint-MLP models (approach I): pixels and skeleton fused late.

Idea: the keypoint MLP is precise when the hand is visible; the crop CNN might recognise the
"arm bent down, hand hidden" appearance when it is not. Each branch produces an embedding, the
embeddings are concatenated and a small classifier decides.

* :class:`HybridCrop2D` — a 2D CNN encodes **each** of the T grayscale desk crops separately
  (shared weights), giving T embeddings; the keypoint vector of the whole window goes through an
  MLP; classifier input = (T + 1) x ``feat_dim``. Two versions were trained:
  ``variant="base"`` — 2 conv blocks (16, 32), CNN head 256 → 128, MLP 256-128-128,
  classifier 256-128-2;
  ``variant="deep"`` — 3 conv blocks (16, 32, 64), CNN head 256-128-32, MLP 256-128-64-32,
  classifier 256-128-64-2.
* :class:`Hybrid3D` — a Conv3d encoder (8, 16 filters, spatial-only pooling) sees the T crops as
  one volume (one embedding of 32), MLP 128-64-32 on the keypoints, classifier 256-64-32-2.

As in the originals, the head has ``n_classes = 2`` outputs trained with cross-entropy (the training
loop handles both 1-logit and 2-class heads). Crop size is configurable (original 64x64); the
original keypoint vector was the time-major window plus 4 position values (104 for 5 frames), so
``kp_dim`` is whatever the data config produces.
"""

from __future__ import annotations

import torch
from torch import nn

HYBRID_VARIANTS = ("base", "deep")


def _mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class CropEncoder2D(nn.Module):
    """Per-frame 2D CNN: (B, T, 1, H, W) → (B, T, feat_dim)."""

    def __init__(self, crop_size: int, conv_channels: tuple[int, ...], head: list[int]) -> None:
        super().__init__()
        factor = 2 ** len(conv_channels)
        if crop_size % factor:
            raise ValueError(f"crop_size must be divisible by {factor}")
        blocks: list[nn.Module] = []
        prev = 1
        for c in conv_channels:
            blocks += [nn.Conv2d(prev, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            prev = c
        self.conv = nn.Sequential(*blocks, nn.Flatten())
        side = crop_size // factor
        self.fc = _mlp([prev * side * side, *head])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        z = self.fc(self.conv(x.reshape(b * t, c, h, w)))
        return z.reshape(b, t, -1)


class HybridCrop2D(nn.Module):
    """``forward(crops (B, T, 1, H, W), keypoints (B, kp_dim))`` → (B, n_classes) logits."""

    def __init__(
        self,
        frames: int = 5,
        kp_dim: int = 104,
        crop_size: int = 64,
        variant: str = "deep",
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        if variant not in HYBRID_VARIANTS:
            raise ValueError(f"variant must be one of {HYBRID_VARIANTS}")
        if variant == "base":
            feat = 128
            self.cnn = CropEncoder2D(crop_size, (16, 32), [256, feat])
            self.kpt_mlp = _mlp([kp_dim, 256, 128, feat])
            self.classifier = _mlp([feat * (frames + 1), 256, 128, n_classes])
        else:
            feat = 32
            self.cnn = CropEncoder2D(crop_size, (16, 32, 64), [256, 128, feat])
            self.kpt_mlp = _mlp([kp_dim, 256, 128, 64, feat])
            self.classifier = _mlp([feat * (frames + 1), 256, 128, 64, n_classes])

    def forward(self, crops: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        crop_feat = self.cnn(crops).flatten(1)
        return self.classifier(torch.cat([crop_feat, self.kpt_mlp(keypoints)], dim=-1))


class Hybrid3D(nn.Module):
    """3D-CNN crop encoder + keypoint MLP; same call signature as :class:`HybridCrop2D`."""

    def __init__(
        self,
        frames: int = 5,
        kp_dim: int = 104,
        crop_size: int = 64,
        crop_feat_dim: int = 32,
        kpt_feat_dim: int = 32,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        if crop_size % 4:
            raise ValueError("crop_size must be divisible by 4")
        side = crop_size // 4
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Flatten(),
            nn.Linear(16 * frames * side * side, 128),
            nn.ReLU(),
            nn.Linear(128, crop_feat_dim),
        )
        self.kpt = _mlp([kp_dim, 128, 64, kpt_feat_dim])
        self.classifier = _mlp([crop_feat_dim + kpt_feat_dim, 256, 64, 32, n_classes])

    def forward(self, crops: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        vol = crops.permute(0, 2, 1, 3, 4)  # (B, T, 1, H, W) -> (B, 1, T, H, W)
        return self.classifier(torch.cat([self.cnn(vol), self.kpt(keypoints)], dim=-1))
