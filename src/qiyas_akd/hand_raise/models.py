"""Hand-raise model definitions: keypoint MLPs, random forest, hybrid image + keypoint model.

**Keras → PyTorch port.** The keypoint MLPs were originally written in Keras; they are
re-implemented here in PyTorch with the same layer sizes, activations and dropout so the whole repo
shares one deep learning stack. Two intentional differences: the models return **logits** (train
with ``BCEWithLogitsLoss``; apply ``sigmoid`` for probabilities) instead of ending in a sigmoid
layer, which is numerically safer; and layers use PyTorch's default initialisation rather than
Keras' Glorot-uniform. The hybrid model was already PyTorch; it also returns logits now.

**Accuracy bug (hybrid training loop).** The four-person hybrid predicted four sigmoid outputs per
image, counted correct *elements* ``((out > 0.5) == labels).sum()`` but divided by the *batch size*,
so the reported "accuracy" could reach 4.0 and was not comparable to anything.
``multilabel_accuracy`` divides by the number of label elements.

**Random-forest pitfalls.** See ``data.py``: a whitespace header created a phantom third class and
made the recall-scored grid search return ``NaN`` for every candidate. ``grid_search_random_forest``
validates binary labels, uses explicit ``pos_label=1`` scorers, raises on scoring errors instead of
recording ``NaN``, and cross-validates with ``GroupKFold`` by video — the original shuffled
``KFold`` put near-identical consecutive frames of one video into both training and validation
folds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from torch import nn

from ..keypoints import NUM_KEYPOINTS
from .data import validate_binary_labels

# Grid of the original random-forest search (180 candidates).
RF_PARAM_GRID: dict[str, list] = {
    "n_estimators": [50, 75, 100],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5],
    "random_state": [42, 35],
}

_SCORERS = {
    "recall": recall_score,
    "precision": precision_score,
    "f1": f1_score,
}


class KeypointMLP(nn.Module):
    """Single-input MLP: ``in → 128 → 64 → [dropout] → 32 → [dropout] → 1`` (ReLU, dropout 0.3).

    Same placement of dropout as the original (none after the first layer).
    """

    def __init__(self, input_dim: int = 2 * NUM_KEYPOINTS, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class TwoBranchMLP(nn.Module):
    """Two-branch MLP: x-coordinates and y-coordinates get separate first layers.

    ``x (10) → 64 → dropout``, ``y (10) → 64 → dropout``, concatenated → 128 → dropout → 64 →
    dropout → 1 (ReLU, dropout 0.3). Each coordinate axis gets its own first layer, and the two
    representations are only mixed after concatenation.

    ``forward(features)`` takes the shared ``[x block, y block]`` feature matrix ``(B, 20)``;
    ``forward(x, y)`` takes the two blocks separately, like the original two-input Keras model.
    """

    def __init__(self, n_keypoints: int = NUM_KEYPOINTS, dropout: float = 0.3) -> None:
        super().__init__()
        self.n_keypoints = n_keypoints

        def branch() -> nn.Sequential:
            return nn.Sequential(nn.Linear(n_keypoints, 64), nn.ReLU(), nn.Dropout(dropout))

        self.x_branch = branch()
        self.y_branch = branch()
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if y is None:
            if x.shape[-1] != 2 * self.n_keypoints:
                raise ValueError(
                    f"expected {2 * self.n_keypoints} features ([x block, y block]); "
                    f"got {x.shape[-1]} — engineered features are not supported by this model"
                )
            x, y = x[..., : self.n_keypoints], x[..., self.n_keypoints :]
        return self.head(torch.cat([self.x_branch(x), self.y_branch(y)], dim=-1))


class HybridModel(nn.Module):
    """EfficientNet-B0 image features (1280-d) fused with a keypoint MLP (→ 64-d).

    Head: ``1344 → 256 → 128 → 64 → dropout 0.3 → num_outputs`` (ReLU). Two configurations existed:

    * four-person frame model: ``keypoint_dim=80`` (4 × 20), ``num_outputs=4`` (multi-label);
    * single-person crop model: ``keypoint_dim=20``, ``num_outputs=1`` (the default).

    Expects ImageNet-normalised ``(B, 3, 224, 224)`` images. torchvision is imported lazily so the
    rest of the package works without it. ``pretrained=True`` loads ImageNet weights (downloads
    them); the default ``False`` builds the architecture only, which is what tests use.
    """

    def __init__(
        self,
        keypoint_dim: int = 2 * NUM_KEYPOINTS,
        num_outputs: int = 1,
        pretrained: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError(
                "HybridModel needs torchvision (pip install torchvision); the keypoint-only models "
                "do not"
            ) from exc
        self.image_model = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        image_dim = self.image_model.classifier[-1].in_features
        self.image_model.classifier = nn.Identity()
        self.keypoints_fc = nn.Sequential(nn.Linear(keypoint_dim, 64), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(image_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_outputs),
        )

    def forward(self, image: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([self.image_model(image), self.keypoints_fc(keypoints)], dim=1)
        return self.fc(feats)


def multilabel_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5
) -> float:
    """Element-wise accuracy over every (sample, output) pair — always in ``[0, 1]``.

    Fix for the original loop, which divided the element count by the batch size.
    """
    if logits.shape != labels.shape:
        raise ValueError(
            f"shape mismatch: logits {tuple(logits.shape)} vs labels {tuple(labels.shape)}"
        )
    if labels.numel() == 0:
        raise ValueError("empty batch")
    pred = torch.sigmoid(logits) > threshold
    return float((pred == labels.bool()).float().mean().item())


def make_random_forest(
    n_estimators: int = 10, random_state: int = 42, **kwargs: object
) -> RandomForestClassifier:
    """Random forest with the original training script's defaults (10 trees, seed 42)."""
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, **kwargs)


def grid_search_random_forest(
    X: np.ndarray,
    y: Sequence[int] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    param_grid: Mapping[str, list] | None = None,
    scoring: str = "recall",
    n_splits: int = 5,
    n_jobs: int | None = None,
) -> GridSearchCV:
    """Grid search over random forests with video-grouped folds and a safe binary scorer.

    ``scoring`` is ``"recall"`` (the original target), ``"precision"`` or ``"f1"``, always computed
    for the positive class ``1``. Returns the fitted ``GridSearchCV``.
    """
    if scoring not in _SCORERS:
        raise ValueError(f"scoring must be one of {sorted(_SCORERS)}")
    y = validate_binary_labels(y, "y")
    groups = np.asarray(groups)
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"need at least {n_splits} groups (videos) for {n_splits}-fold GroupKFold")
    scorer = make_scorer(_SCORERS[scoring], pos_label=1, zero_division=0)
    search = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=dict(param_grid if param_grid is not None else RF_PARAM_GRID),
        scoring=scorer,
        cv=GroupKFold(n_splits=n_splits),
        error_score="raise",
        n_jobs=n_jobs,
    )
    search.fit(np.asarray(X), y, groups=groups)
    return search
