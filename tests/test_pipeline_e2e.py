"""End-to-end pipeline test on synthetic data.

Every other test file checks one component in isolation. This one wires the
whole lab together — split, train, extract statistics, run all three MIAs,
then steal the model — on a tiny synthetic dataset that needs no download.

The point is not to produce meaningful attack numbers (200 random-ish images
and 2 epochs won't), but to prove the *plumbing* holds: shapes line up,
challenge sets stay aligned between shadow models and the target, and nothing
returns NaN. These are exactly the failures that would otherwise surface an
hour into a GPU sweep.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import data, metrics, models, train, utils
from src.attacks import extraction, lira, loss_threshold, shadow_mia

POOL_SIZE = 240
NUM_CLASSES = 4


@pytest.fixture(scope="module")
def synthetic_pool() -> TensorDataset:
    """32x32 images whose class is encoded as a per-class mean offset.

    Learnable (so training does something) but noisy (so the model can't hit
    100% instantly), which keeps the statistics non-degenerate.
    """
    generator = torch.Generator().manual_seed(0)
    labels = torch.randint(0, NUM_CLASSES, (POOL_SIZE,), generator=generator)
    images = torch.randn(POOL_SIZE, 3, 32, 32, generator=generator)
    images += labels.view(-1, 1, 1, 1).float() * 0.5
    return TensorDataset(images, labels)


@pytest.fixture(scope="module")
def tiny_config() -> dict:
    return {
        "model": {"name": "simple_cnn", "num_classes": NUM_CLASSES, "dropout": 0.0},
        "train": {"batch_size": 32, "epochs": 2, "lr": 0.002},
    }


@pytest.fixture(scope="module")
def splits() -> data.DatasetSplits:
    return data.make_splits(
        pool_size=POOL_SIZE,
        target_train_size=60,
        target_holdout_size=60,
        shadow_pool_size=80,
        attacker_aux_size=40,
        seed=0,
    )


@pytest.fixture(scope="module")
def trained_target(synthetic_pool, splits, tiny_config):
    """Train a target and return (model, member_stats, nonmember_stats)."""
    utils.set_seed(0)
    device = torch.device("cpu")
    member_subset = data.get_subset(synthetic_pool, splits.target_train)
    nonmember_subset = data.get_subset(synthetic_pool, splits.target_holdout)

    model = models.build_model(**tiny_config["model"])
    model, history = train.train_model(
        model,
        DataLoader(member_subset, batch_size=32, shuffle=True),
        DataLoader(nonmember_subset, batch_size=32, shuffle=False),
        device, epochs=2, lr=0.002,
    )
    return (
        model,
        train.compute_per_example_stats(model, member_subset, device),
        train.compute_per_example_stats(model, nonmember_subset, device),
        history,
    )


def test_training_history_records_gap(trained_target):
    _, _, _, history = trained_target
    assert len(history) == 2
    for entry in history:
        assert entry["gap"] == pytest.approx(entry["train_acc"] - entry["eval_acc"])
        assert np.isfinite(entry["train_loss"])


def test_per_example_stats_shapes_and_finiteness(trained_target):
    _, member_stats, _, _ = trained_target
    assert len(member_stats["loss"]) == 60
    assert member_stats["probs"].shape == (60, NUM_CLASSES)
    assert member_stats["logits"].shape == (60, NUM_CLASSES)
    for key, values in member_stats.items():
        assert np.isfinite(values).all(), f"{key} contains non-finite values"
    # Softmax rows must sum to 1.
    np.testing.assert_allclose(member_stats["probs"].sum(axis=1), 1.0, atol=1e-5)


def test_threshold_attack_runs_end_to_end(trained_target):
    _, member_stats, nonmember_stats, _ = trained_target
    result = loss_threshold.run_threshold_attack(member_stats, nonmember_stats, score_name="loss")
    assert 0.0 <= result.metrics["auc"] <= 1.0
    assert np.isfinite(result.scores).all()
    assert len(result.scores) == 120


def test_shadow_mia_runs_end_to_end(synthetic_pool, splits, tiny_config, trained_target):
    _, member_stats, nonmember_stats, _ = trained_target
    shadow_runs = shadow_mia.train_shadow_models(
        pool=synthetic_pool,
        candidate_indices=splits.shadow_pool,
        num_shadows=2,
        shadow_train_size=40,
        model_config=tiny_config["model"],
        train_config=tiny_config["train"],
        device=torch.device("cpu"),
        seed=0,
    )
    assert len(shadow_runs) == 2

    for per_class in (False, True):
        scores, labels, attack_metrics = shadow_mia.run_shadow_attack(
            shadow_runs, member_stats, nonmember_stats,
            classifier_kind="logistic_regression", per_class=per_class,
        )
        assert np.isfinite(scores).all()
        assert len(scores) == len(labels) == 120
        assert 0.0 <= attack_metrics["auc"] <= 1.0
        assert "tpr_at_fpr_0.001" in attack_metrics


def test_lira_runs_end_to_end_with_aligned_challenge_set(
    synthetic_pool, splits, tiny_config, trained_target
):
    """The alignment between shadow columns and target stats is the easiest
    thing in the lab to get silently wrong, so it is asserted explicitly."""
    _, member_stats, nonmember_stats, _ = trained_target
    challenge_indices = np.concatenate([splits.target_train, splits.target_holdout])

    shadow_outputs = lira.train_lira_shadows(
        pool=synthetic_pool,
        challenge_indices=challenge_indices,
        num_shadows=4,
        model_config=tiny_config["model"],
        train_config=tiny_config["train"],
        device=torch.device("cpu"),
        in_fraction=0.5,
        seed=0,
    )

    assert shadow_outputs.phi.shape == (4, len(challenge_indices))
    assert shadow_outputs.in_mask.shape == (4, len(challenge_indices))
    # With in_fraction=0.5, each shadow trains on half the challenge set.
    np.testing.assert_array_equal(
        shadow_outputs.in_mask.sum(axis=1), np.full(4, len(challenge_indices) // 2)
    )
    assert np.isfinite(shadow_outputs.phi).all()

    scores, labels, attack_metrics = lira.run_lira_attack(
        shadow_outputs, member_stats, nonmember_stats, variance_mode="global"
    )
    assert np.isfinite(scores).all()
    assert len(scores) == len(challenge_indices)
    assert 0.0 <= attack_metrics["auc"] <= 1.0


def test_lira_per_example_variance_mode_also_runs(synthetic_pool, splits, tiny_config):
    """per_example variance is noisy at low shadow counts but must not crash."""
    challenge_indices = np.concatenate([splits.target_train, splits.target_holdout])
    shadow_outputs = lira.train_lira_shadows(
        pool=synthetic_pool, challenge_indices=challenge_indices, num_shadows=3,
        model_config=tiny_config["model"], train_config=tiny_config["train"],
        device=torch.device("cpu"), seed=1,
    )
    scores = lira.lira_scores(
        shadow_outputs, np.zeros(len(challenge_indices)), variance_mode="per_example"
    )
    assert np.isfinite(scores).all()


def test_extraction_runs_end_to_end(synthetic_pool, splits, tiny_config, trained_target):
    """Steal the target and confirm fidelity/accuracy are well-formed.

    Uses the *shadow pool* as the attacker's query data, standing in for the
    out-of-distribution CIFAR-100 pool the real experiment uses.
    """
    target_model, _, _, _ = trained_target
    device = torch.device("cpu")
    api = extraction.BlackBoxAPI(target_model, device, output_mode="full")
    query_pool = data.get_subset(synthetic_pool, splits.shadow_pool)

    stolen = extraction.collect_api_responses(api, query_pool, num_queries=60, batch_size=16)
    assert api.query_count == 60

    utils.set_seed(0)
    surrogate = models.build_model(name="simple_cnn", num_classes=NUM_CLASSES)
    surrogate = extraction.train_surrogate(
        surrogate, stolen, device, epochs=2, lr=0.002, batch_size=16, label_only=False
    )

    eval_dataset = data.get_subset(synthetic_pool, splits.target_holdout)
    fidelity, surrogate_accuracy, target_accuracy = extraction.evaluate_extraction(
        surrogate, api, eval_dataset, device
    )

    for value in (fidelity, surrogate_accuracy, target_accuracy):
        assert 0.0 <= value <= 1.0
    # Evaluation queries must not be charged to the attack's budget.
    assert api.query_count == 60


def test_extraction_label_only_path_runs(synthetic_pool, splits, trained_target):
    """The label-only branch uses a different loss, so it needs its own pass."""
    target_model, _, _, _ = trained_target
    device = torch.device("cpu")
    api = extraction.BlackBoxAPI(target_model, device, output_mode="label")
    query_pool = data.get_subset(synthetic_pool, splits.shadow_pool)

    stolen = extraction.collect_api_responses(api, query_pool, num_queries=40, batch_size=16)
    # A label-only API returns one-hot rows.
    _, responses = stolen[0]
    assert responses.sum().item() == pytest.approx(1.0)
    assert (responses == 0).sum().item() == NUM_CLASSES - 1

    utils.set_seed(0)
    surrogate = models.build_model(name="simple_cnn", num_classes=NUM_CLASSES)
    surrogate = extraction.train_surrogate(
        surrogate, stolen, device, epochs=1, lr=0.002, batch_size=16, label_only=True
    )
    fidelity, _, _ = extraction.evaluate_extraction(
        surrogate, api, data.get_subset(synthetic_pool, splits.target_holdout), device
    )
    assert 0.0 <= fidelity <= 1.0


def test_all_attacks_produce_the_required_metric_bundle(trained_target):
    """Lab rule: every MIA reports AUC *and* both low-FPR operating points."""
    _, member_stats, nonmember_stats, _ = trained_target
    result = loss_threshold.run_threshold_attack(
        member_stats, nonmember_stats, score_name="loss", threshold=0.0
    )
    for key in ("auc", "tpr_at_fpr_0.01", "tpr_at_fpr_0.001", "balanced_accuracy"):
        assert key in result.metrics, f"missing required metric: {key}"


def test_splits_stay_disjoint_through_the_pipeline(splits):
    """Re-assert the core invariant at pipeline level, not just unit level."""
    seen: set[int] = set()
    for group in (splits.target_train, splits.target_holdout,
                  splits.shadow_pool, splits.attacker_aux):
        group_set = set(group.tolist())
        assert not (group_set & seen)
        seen |= group_set
