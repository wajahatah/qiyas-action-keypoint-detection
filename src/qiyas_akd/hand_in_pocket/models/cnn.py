"""Image-based temporal models (approach H): grayscale frames stacked over time.

The motivation was that keypoints alone cannot see a hand *inside* a pocket — once occluded, the
hand keypoint is just missing — so pixels might carry what the skeleton loses.

* :class:`TemporalGrayCNN` — T grayscale images stacked as **channels** of one 2D CNN (4 conv blocks
  16-32-64-128, each 3x3 + ReLU + 2x2 max-pool; head 256-128-64-Dropout 0.3-1). Originally run on
  640x640 full frames.
* :class:`TemporalCrop3DCNN` — desk-ROI crops as a (1, T, H, W) volume through a small **Conv3d**
  (8 and 16 filters, 3x3x3, pooling only spatially so the time axis is preserved; head 128-64-1).

Both lost to the keypoint MLP on the bench: with tens of thousands of near-duplicate frames from a
few cameras, image models overfit to backgrounds and desk appearance. Image size is a constructor
argument (the head size is computed from it) so tests can run on 32x32 inputs. Outputs are logits.
"""

from __future__ import annotations

import torch
from torch import nn


def _check_size(img_size: int, factor: int) -> None:
    if img_size % factor:
        raise ValueError(f"img_size must be divisible by {factor}, got {img_size}")


class TemporalGrayCNN(nn.Module):
    """Input (B, T, H, W) in [0, 1] → (B, 1) logits."""

    def __init__(self, frames: int = 5, img_size: int = 640, dropout: float = 0.3) -> None:
        super().__init__()
        _check_size(img_size, 16)
        chans = [frames, 16, 32, 64, 128]
        blocks: list[nn.Module] = []
        for a, b in zip(chans[:-1], chans[1:], strict=True):
            blocks += [nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        self.conv = nn.Sequential(*blocks)
        side = img_size // 16
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * side * side, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class TemporalCrop3DCNN(nn.Module):
    """Input (B, T, H, W) desk crops in [0, 1] → (B, 1) logits."""

    def __init__(self, frames: int = 5, img_size: int = 64) -> None:
        super().__init__()
        _check_size(img_size, 4)
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
        )
        side = img_size // 4
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * frames * side * side, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x.unsqueeze(1)))
