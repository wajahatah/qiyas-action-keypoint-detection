"""MLP architectures c0..c4 for flattened keypoint windows, exactly as they were compared.

Layer stacks recovered from the training/inference scripts' history (the arch code was the suffix of
each model file name):

* **c0** — ``in → 64 → 32 → 1``, ReLU, Dropout 0.3 after each hidden layer. The first and final
  architecture: it became the deployed 3-frame model.
* **c1** — ``in → 64 → 32 → 16 → 1``, ReLU, Dropout 0.3 after the 32- and 16-unit layers.
* **c2** — ``in → 128 → 64 → 32 → 1``, ReLU, Dropout 0.3 after the 64- and 32-unit layers.
* **c3** — ``in → 128 → 64 → 32 → 16 → 1``, ReLU, Dropout 0.3 after the 64- and 32-unit layers.
* **c4** — ``in → 64 → 32 → 1``, ReLU, no dropout.

Deeper/wider variants did not beat c0 on the 14-video bench, i.e. capacity was not the bottleneck —
the dataset design (window, skipping, labelling, balancing) was.

All variants return **logits** (no final sigmoid). The early scripts ended in ``Sigmoid`` +
``BCELoss``; the later recipe uses ``BCEWithLogitsLoss``, which is numerically stabler and
mathematically the same model, so ``torch.sigmoid(logits)`` reproduces the original probabilities.
"""

from __future__ import annotations

from torch import nn

# (hidden widths, indices of hidden layers followed by Dropout(0.3))
MLP_ARCHS: dict[str, tuple[tuple[int, ...], frozenset[int]]] = {
    "c0": ((64, 32), frozenset({0, 1})),
    "c1": ((64, 32, 16), frozenset({1, 2})),
    "c2": ((128, 64, 32), frozenset({1, 2})),
    "c3": ((128, 64, 32, 16), frozenset({1, 2})),
    "c4": ((64, 32), frozenset()),
}


class WindowMLP(nn.Module):
    """Binary classifier over one flattened window (``in_dim`` = steps x 20 [+ extras] + 4)."""

    def __init__(self, in_dim: int, arch: str = "c0", dropout: float = 0.3) -> None:
        super().__init__()
        if arch not in MLP_ARCHS:
            raise ValueError(f"unknown MLP arch {arch!r}; choose from {sorted(MLP_ARCHS)}")
        widths, drop_after = MLP_ARCHS[arch]
        layers: list[nn.Module] = []
        prev = in_dim
        for i, w in enumerate(widths):
            layers += [nn.Linear(prev, w), nn.ReLU()]
            if i in drop_after:
                layers.append(nn.Dropout(dropout))
            prev = w
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.arch = arch

    def forward(self, x):  # (B, in_dim) -> (B, 1) logits
        return self.net(x)
