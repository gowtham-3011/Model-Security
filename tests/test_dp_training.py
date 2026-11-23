"""Tests that DP-SGD actually runs and that the models are Opacus-compatible.

These are deliberately tiny — one step on synthetic data. The point is to
catch the failure mode where DP training is misconfigured and silently
falls over (or, worse, silently trains *without* privacy) long before
committing an hour of GPU time to the epsilon sweep.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import defenses, models, train

opacus = pytest.importorskip("opacus", reason="opacus is required for DP-SGD tests")


def _tiny_image_loader(num_examples: int = 32, batch_size: int = 8) -> DataLoader:
    """Synthetic 32x32 RGB data — no dataset download needed."""
    generator = torch.Generator().manual_seed(0)
    images = torch.randn(num_examples, 3, 32, 32, generator=generator)
    labels = torch.randint(0, 10, (num_examples,), generator=generator)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=True)


def test_simple_cnn_is_opacus_compatible():
    """Our models must pass Opacus validation with no complaints.

    This is what using GroupNorm instead of BatchNorm buys us; if someone
    swaps a BatchNorm back in, this test is the tripwire.
    """
    assert defenses.validate_dp_compatible(models.SimpleCNN()) == []


def test_small_resnet_is_opacus_compatible():
    assert defenses.validate_dp_compatible(models.SmallResNet()) == []


def test_tabular_mlp_is_opacus_compatible():
    assert defenses.validate_dp_compatible(models.TabularMLP(input_dim=20)) == []


def test_batchnorm_model_is_rejected_with_helpful_error():
    """A BatchNorm model must fail loudly, with guidance, not mysteriously."""
    bad_model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8),
                              nn.Flatten(), nn.Linear(8 * 32 * 32, 10))
    assert defenses.validate_dp_compatible(bad_model) != []

    loader = _tiny_image_loader()
    with pytest.raises(ValueError, match="not compatible with Opacus"):
        defenses.make_private(
            model=bad_model,
            optimizer=torch.optim.Adam(bad_model.parameters()),
            data_loader=loader,
            epochs=1,
            target_epsilon=8.0,
            target_delta=1e-3,
        )


def test_make_private_returns_positive_noise_multiplier():
    model = models.SimpleCNN()
    loader = _tiny_image_loader()
    _, private_optimizer, _, _ = defenses.make_private(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        data_loader=loader,
        epochs=1,
        target_epsilon=8.0,
        target_delta=1e-3,
    )
    # Noise must actually be added, or there is no privacy despite the label.
    assert private_optimizer.noise_multiplier > 0


def test_tighter_epsilon_requires_more_noise():
    """A smaller privacy budget must translate into a larger noise multiplier.

    If this inverts, the epsilon sweep is meaningless — so it's worth pinning.
    """
    noise_by_epsilon = {}
    for epsilon in (1.0, 20.0):
        model = models.SimpleCNN()
        _, private_optimizer, _, _ = defenses.make_private(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            data_loader=_tiny_image_loader(),
            epochs=1,
            target_epsilon=epsilon,
            target_delta=1e-3,
        )
        noise_by_epsilon[epsilon] = private_optimizer.noise_multiplier

    assert noise_by_epsilon[1.0] > noise_by_epsilon[20.0]


def test_dp_training_runs_one_epoch_and_spends_budget():
    """DP training completes and reports a finite epsilon within the target."""
    model = models.SimpleCNN()
    loader = _tiny_image_loader()

    trained, history, privacy_spent = train.train_model_dp(
        model, loader, loader, torch.device("cpu"),
        epochs=1, lr=1e-3, target_epsilon=8.0, target_delta=1e-3,
    )

    assert len(history) == 1
    assert np.isfinite(privacy_spent["epsilon"])
    # The accountant must not overshoot the requested budget.
    assert privacy_spent["epsilon"] <= 8.0 + 1e-6
    assert privacy_spent["noise_multiplier"] > 0
    # A plain nn.Module must come back, not Opacus's GradSampleModule wrapper,
    # so checkpoints and attack code see the original parameter names.
    assert not hasattr(trained, "_module")
    assert trained(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_dp_training_with_infinite_epsilon_skips_dp():
    """epsilon=inf must fall through to ordinary training, so the sweep can
    treat "no DP" as just another point on the axis."""
    model = models.SimpleCNN()
    loader = _tiny_image_loader()

    _, history, privacy_spent = train.train_model_dp(
        model, loader, loader, torch.device("cpu"),
        epochs=1, lr=1e-3, target_epsilon=float("inf"), target_delta=1e-3,
    )

    assert len(history) == 1
    assert privacy_spent["epsilon"] == float("inf")
    assert "noise_multiplier" not in privacy_spent


def test_delta_for_dataset_size():
    assert defenses.delta_for_dataset_size(8000) == pytest.approx(1 / 8000)
    with pytest.raises(ValueError):
        defenses.delta_for_dataset_size(0)
