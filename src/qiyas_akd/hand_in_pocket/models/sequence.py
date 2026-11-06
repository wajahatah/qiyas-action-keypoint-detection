"""LSTM / GRU over a window treated as a sequence of per-frame vectors (approach F, abandoned).

Each step is 24 values: ``x`` of the 10 keypoints, then their ``y``, then the 4 desk position values
(repeated every step) — the per-step layout of the original sequence export. A 5-frame window is
therefore a (5, 24) sequence. Original sizes: LSTM 2 layers x 64, GRU 3 layers x 64, classifier on
the last step's hidden state.

The LSTM experiment replaced missing values (-1) with -10 and clamped inputs to [-10, 1], to push
"missing" far from any real coordinate; ``missing_value`` / ``clamp`` keep that option. Neither
model beat the flat MLP: with only 3-5 steps there is little sequence structure for a recurrent
model to exploit, and the MLP sees all steps at once anyway.

Outputs are logits (see ``mlp`` for why); the originals ended in a sigmoid.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from ...keypoints import MISSING, NUM_KEYPOINTS


def sequence_columns(time_names: Sequence[str], static_cols: Sequence[str]) -> list[list[str]]:
    """Per step, the 24 column names in sequence order (x0..x9, y0..y9, position_a..d)."""
    return [
        [f"kp_{k}_x_{t}" for k in range(NUM_KEYPOINTS)]
        + [f"kp_{k}_y_{t}" for k in range(NUM_KEYPOINTS)]
        + list(static_cols)
        for t in time_names
    ]


def to_sequences(
    windows: pd.DataFrame,
    time_names: Sequence[str],
    static_cols: Sequence[str],
    missing_value: float = MISSING,
) -> np.ndarray:
    """Windowed table → float32 (N, T, 20 + len(static_cols)), whatever the column layout."""
    steps = sequence_columns(time_names, static_cols)
    arr = np.stack([windows[c].to_numpy(np.float32) for c in steps], axis=1)
    if missing_value != MISSING:
        arr[arr == MISSING] = missing_value
    return arr


class _Recurrent(nn.Module):
    cell: type[nn.Module]

    def __init__(
        self,
        input_size: int = 24,
        hidden_size: int = 64,
        num_layers: int = 2,
        clamp: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.rnn = self.cell(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.clamp = tuple(clamp) if clamp is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F) -> (B, 1) logits
        if self.clamp is not None:
            x = torch.clamp(x, self.clamp[0], self.clamp[1])
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class LSTMClassifier(_Recurrent):
    """2 x 64 LSTM by default (original: missing → -10, inputs clamped to [-10, 1])."""

    cell = nn.LSTM


class GRUClassifier(_Recurrent):
    """3 x 64 GRU by default."""

    cell = nn.GRU

    def __init__(
        self,
        input_size: int = 24,
        hidden_size: int = 64,
        num_layers: int = 3,
        clamp: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(input_size, hidden_size, num_layers, clamp)
