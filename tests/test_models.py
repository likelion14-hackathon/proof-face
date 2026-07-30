"""Phase 2 scaffold smoke tests.

Skipped entirely when the ``dl`` extra (torch/timm/albumentations) is absent, so
the core Phase 1 suite stays dependency-light.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("albumentations")

from skin_metrics.models.dataset import (  # noqa: E402
    PHYSICS_DIM,
    TARGET_NAMES,
    DummyLabelGenerator,
    SkinDataset,
)


def test_dummy_generator_shapes():
    gen = DummyLabelGenerator(image_size=32, seed=0)
    samples = gen.build(5)
    assert len(samples) == 5
    s = samples[0]
    assert s.physics.shape == (PHYSICS_DIM,)
    assert s.targets.shape == (len(TARGET_NAMES),)
    assert 1 <= s.fitzpatrick <= 6
    assert (s.targets >= 0).all() and (s.targets <= 100).all()


def test_dataset_item_tensors():
    ds = SkinDataset.dummy(n=4, image_size=32, train=True, seed=1)
    item = ds[0]
    assert item["image"].shape == (3, 32, 32)
    assert item["physics"].shape == (PHYSICS_DIM,)
    assert item["targets"].shape == (len(TARGET_NAMES),)


def test_network_forward_shapes():
    from skin_metrics.models.network import SkinNet, uncertainty_weighted_loss

    model = SkinNet(pretrained=False)
    image = torch.randn(2, 3, 64, 64)
    physics = torch.randn(2, PHYSICS_DIM)
    out = model(image, physics, grl_lambda=0.5)
    assert out["predictions"].shape == (2, len(TARGET_NAMES))
    assert out["domain_logits"].shape[0] == 2

    loss = uncertainty_weighted_loss(torch.tensor([1.0, 2.0, 3.0]), model.log_vars)
    assert torch.isfinite(loss)


def test_grad_reverse_negates_gradient():
    from skin_metrics.models.network import grad_reverse

    x = torch.ones(3, requires_grad=True)
    y = grad_reverse(x, 2.0).sum()
    y.backward()
    assert torch.allclose(x.grad, torch.full((3,), -2.0))


def test_training_loop_regression_runs():
    from skin_metrics.models.train import run_training

    result = run_training(
        use_dummy=True, mode="regression", epochs=1, n_dummy=32, image_size=32, batch_size=8
    )
    assert len(result["train_losses"]) == 1
    assert np.isfinite(result["train_losses"][0])
    assert "overall" in result["val_report"]
    assert "by_fitzpatrick" in result["val_report"]


def test_training_loop_ranking_runs():
    from skin_metrics.models.train import run_training

    result = run_training(
        use_dummy=True, mode="ranking", epochs=1, n_dummy=32, image_size=32, batch_size=8
    )
    assert np.isfinite(result["train_losses"][0])
    assert result["mode"] == "ranking"
