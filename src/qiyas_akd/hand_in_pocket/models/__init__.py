"""Model zoo for hand in pocket, plus a registry so configs can name a model by ``kind``.

``INPUT_KIND`` tells the training/inference code what each torch model consumes:

* ``tabular`` — the flat feature vector (``DataConfig.feature_columns``);
* ``sequence`` — (T, 24) per-step vectors (``sequence.to_sequences``);
* ``image`` — (T, H, W) grayscale crops;
* ``image+kp`` — crops (T, 1, H, W) and the flat keypoint vector.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from .cnn import TemporalCrop3DCNN, TemporalGrayCNN
from .hybrid import Hybrid3D, HybridCrop2D
from .mlp import MLP_ARCHS, WindowMLP
from .sequence import GRUClassifier, LSTMClassifier
from .trees import RF_PARAM_GRID, TREE_KINDS, make_tree_model

TORCH_MODELS: dict[str, type[nn.Module]] = {
    "mlp": WindowMLP,
    "lstm": LSTMClassifier,
    "gru": GRUClassifier,
    "temporal_cnn": TemporalGrayCNN,
    "temporal_crop_cnn3d": TemporalCrop3DCNN,
    "hybrid2d": HybridCrop2D,
    "hybrid3d": Hybrid3D,
}

INPUT_KIND: dict[str, str] = {
    "mlp": "tabular",
    "lstm": "sequence",
    "gru": "sequence",
    "temporal_cnn": "image",
    "temporal_crop_cnn3d": "image",
    "hybrid2d": "image+kp",
    "hybrid3d": "image+kp",
}


def build_torch_model(kind: str, **params: Any) -> nn.Module:
    """Instantiate a torch model by registry name with constructor ``params``."""
    if kind not in TORCH_MODELS:
        raise ValueError(f"unknown torch model {kind!r}; choose from {sorted(TORCH_MODELS)}")
    return TORCH_MODELS[kind](**params)


__all__ = [
    "INPUT_KIND",
    "MLP_ARCHS",
    "RF_PARAM_GRID",
    "TORCH_MODELS",
    "TREE_KINDS",
    "GRUClassifier",
    "Hybrid3D",
    "HybridCrop2D",
    "LSTMClassifier",
    "TemporalCrop3DCNN",
    "TemporalGrayCNN",
    "WindowMLP",
    "build_torch_model",
    "make_tree_model",
]
