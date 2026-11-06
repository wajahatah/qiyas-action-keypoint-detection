"""Hand in pocket — model definitions: architecture codes and forward-pass shapes (tiny inputs)."""

import pytest
import torch
from torch import nn

from qiyas_akd.hand_in_pocket.models import (
    MLP_ARCHS,
    TORCH_MODELS,
    GRUClassifier,
    Hybrid3D,
    HybridCrop2D,
    LSTMClassifier,
    TemporalCrop3DCNN,
    TemporalGrayCNN,
    WindowMLP,
    build_torch_model,
    make_tree_model,
)

B, T, S = 3, 5, 32


@pytest.mark.parametrize(
    ("arch", "widths", "n_dropout"),
    [
        ("c0", [64, 32, 1], 2),
        ("c1", [64, 32, 16, 1], 2),
        ("c2", [128, 64, 32, 1], 2),
        ("c3", [128, 64, 32, 16, 1], 2),
        ("c4", [64, 32, 1], 0),
    ],
)
def test_mlp_arch_codes(arch, widths, n_dropout):
    m = WindowMLP(64, arch)
    linears = [x for x in m.net if isinstance(x, nn.Linear)]
    assert [x.out_features for x in linears] == widths
    assert linears[0].in_features == 64
    assert sum(isinstance(x, nn.Dropout) for x in m.net) == n_dropout
    assert not any(isinstance(x, nn.Sigmoid) for x in m.net)  # logits out
    assert m(torch.zeros(B, 64)).shape == (B, 1)


def test_mlp_c0_dropout_positions_match_original():
    layers = list(WindowMLP(104, "c0").net)
    kinds = [type(x).__name__ for x in layers]
    assert kinds == ["Linear", "ReLU", "Dropout", "Linear", "ReLU", "Dropout", "Linear"]
    assert set(MLP_ARCHS) == {"c0", "c1", "c2", "c3", "c4"}
    with pytest.raises(ValueError):
        WindowMLP(64, "c9")


def test_recurrent_shapes():
    x = torch.randn(B, T, 24)
    assert LSTMClassifier(clamp=(-10, 1))(x).shape == (B, 1)
    gru = GRUClassifier()
    assert gru.rnn.num_layers == 3 and gru.rnn.hidden_size == 64
    assert gru(x).shape == (B, 1)


def test_image_model_shapes():
    crops = torch.rand(B, T, S, S)
    assert TemporalGrayCNN(frames=T, img_size=S)(crops).shape == (B, 1)
    assert TemporalCrop3DCNN(frames=T, img_size=S)(crops).shape == (B, 1)
    with pytest.raises(ValueError):
        TemporalGrayCNN(img_size=30)


@pytest.mark.parametrize("variant", ["base", "deep"])
def test_hybrid_2d_shapes(variant):
    m = HybridCrop2D(frames=T, kp_dim=104, crop_size=S, variant=variant)
    out = m(torch.rand(B, T, 1, S, S), torch.rand(B, 104))
    assert out.shape == (B, 2)


def test_hybrid_3d_shape_and_registry():
    m = Hybrid3D(frames=3, kp_dim=64, crop_size=16, n_classes=1)
    assert m(torch.rand(B, 3, 1, 16, 16), torch.rand(B, 64)).shape == (B, 1)
    assert isinstance(build_torch_model("mlp", in_dim=10, arch="c4"), WindowMLP)
    assert set(TORCH_MODELS) >= {"mlp", "gru", "lstm", "hybrid3d"}
    with pytest.raises(ValueError):
        build_torch_model("transformer")


def test_tree_factory_defaults():
    rf = make_tree_model("rf", seed=1)
    assert rf.n_estimators == 100 and rf.random_state == 1
    with pytest.raises(ValueError):
        make_tree_model("svm")
