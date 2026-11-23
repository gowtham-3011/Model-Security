"""Tests for the attack implementations on synthetic per-example statistics.

No model training here — each attack is fed hand-built "model outputs" where
the right answer is known, which isolates attack logic from training noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import metrics, utils
from src.attacks import lira, loss_threshold, shadow_mia


def _make_stats(probs: np.ndarray, true_labels: np.ndarray) -> dict[str, np.ndarray]:
    """Build a stats dict of the shape train.compute_per_example_stats returns."""
    row_indices = np.arange(len(true_labels))
    true_class_probs = probs[row_indices, true_labels]
    return {
        "probs": probs.astype(np.float32),
        # Logits consistent with the probs, up to the additive constant softmax
        # ignores, so LiRA's transform can be checked against them.
        "logits": np.log(np.clip(probs, 1e-12, None)).astype(np.float32),
        "true_label": true_labels,
        "loss": -np.log(np.clip(true_class_probs, 1e-12, None)),
        "confidence_true_label": true_class_probs,
        "max_prob": probs.max(axis=1),
        "predicted_label": probs.argmax(axis=1),
    }


@pytest.fixture
def separable_target_stats():
    """Members confidently correct, non-members not — a clearly leaky model."""
    rng = np.random.default_rng(0)
    num = 200

    member_confidence = rng.uniform(0.95, 0.999, num)
    nonmember_confidence = rng.uniform(0.3, 0.6, num)

    def build(confidences):
        probs = np.zeros((len(confidences), 10))
        probs[:, 0] = confidences
        probs[:, 1:] = ((1 - confidences) / 9)[:, None]
        return _make_stats(probs, np.zeros(len(confidences), dtype=int))

    return build(member_confidence), build(nonmember_confidence)


# ---------------- Phase 2: threshold attack ----------------

def test_loss_scores_are_negated_losses():
    stats = {"loss": np.array([0.1, 2.0])}
    np.testing.assert_allclose(loss_threshold.loss_scores(stats), np.array([-0.1, -2.0]))


def test_threshold_attack_detects_a_leaky_model(separable_target_stats):
    member_stats, nonmember_stats = separable_target_stats
    result = loss_threshold.run_threshold_attack(member_stats, nonmember_stats, score_name="loss")
    # Fully separable by construction -> AUC 1.
    assert result.metrics["auc"] == pytest.approx(1.0)
    assert result.metrics["tpr_at_fpr_0.01"] == pytest.approx(1.0)


def test_threshold_attack_finds_nothing_on_a_private_model():
    """Identical member/non-member distributions -> AUC near chance.

    This is the negative control: if this test ever showed high AUC, the
    attack would be reading a bug rather than leakage.
    """
    rng = np.random.default_rng(1)
    confidences = rng.uniform(0.4, 0.9, 400)
    probs = np.zeros((400, 10))
    probs[:, 0] = confidences
    probs[:, 1:] = ((1 - confidences) / 9)[:, None]
    stats = _make_stats(probs, np.zeros(400, dtype=int))

    halves = ({k: v[:200] for k, v in stats.items()}, {k: v[200:] for k, v in stats.items()})
    result = loss_threshold.run_threshold_attack(*halves, score_name="loss")
    assert result.metrics["auc"] == pytest.approx(0.5, abs=0.08)


def test_confidence_and_loss_scores_are_rank_equivalent():
    """Loss and confidence must produce *identical* ranking metrics.

    Cross-entropy loss is exactly -log(confidence_true_label), so -loss is a
    strictly increasing transform of confidence. Every order-based metric
    (AUC, TPR@FPR) must therefore match exactly — not just approximately, and
    not just within one class. This pins the property down across many
    classes and confidence levels, so that if someone ever changes how `loss`
    is computed and breaks the identity, it fails here loudly.
    """
    rng = np.random.default_rng(2)
    num = 400
    # Random true labels across all 10 classes, and messy (non-uniform)
    # distributions over the remaining mass — the general case.
    true_labels = rng.integers(0, 10, num)
    probs = rng.dirichlet(np.ones(10) * 0.5, size=num)
    stats = _make_stats(probs, true_labels)

    members = {k: v[:200] for k, v in stats.items()}
    nonmembers = {k: v[200:] for k, v in stats.items()}

    by_loss = loss_threshold.run_threshold_attack(members, nonmembers, score_name="loss")
    by_confidence = loss_threshold.run_threshold_attack(members, nonmembers, score_name="confidence")

    assert by_loss.metrics["auc"] == pytest.approx(by_confidence.metrics["auc"], abs=1e-12)
    assert by_loss.metrics["tpr_at_fpr_0.01"] == pytest.approx(
        by_confidence.metrics["tpr_at_fpr_0.01"], abs=1e-12
    )
    assert by_loss.metrics["tpr_at_fpr_0.001"] == pytest.approx(
        by_confidence.metrics["tpr_at_fpr_0.001"], abs=1e-12
    )


def test_choose_threshold_separates_clean_case():
    """With members at 1.0 and non-members at 0.0, the threshold must land between."""
    threshold = loss_threshold.choose_threshold(np.ones(50), np.zeros(50))
    assert 0.0 < threshold < 1.0


def test_unknown_score_name_raises():
    with pytest.raises(ValueError, match="Unknown score_name"):
        loss_threshold.run_threshold_attack({}, {}, score_name="telepathy")


# ---------------- Phase 3: shadow attack features ----------------

def test_attack_features_shape_is_num_classes_plus_two():
    stats = _make_stats(np.full((5, 10), 0.1), np.zeros(5, dtype=int))
    features = shadow_mia.build_attack_features(stats)
    assert features.shape == (5, 12)  # 10 probs + loss + true-label confidence


def test_attack_features_sorting_is_descending():
    probs = np.array([[0.1, 0.7, 0.2]])
    features = shadow_mia.build_attack_features(_make_stats(probs, np.array([0])), sort_probs=True)
    np.testing.assert_allclose(features[0, :3], np.array([0.7, 0.2, 0.1]), atol=1e-6)


def test_attack_features_unsorted_preserves_class_order():
    probs = np.array([[0.1, 0.7, 0.2]])
    features = shadow_mia.build_attack_features(_make_stats(probs, np.array([0])), sort_probs=False)
    np.testing.assert_allclose(features[0, :3], np.array([0.1, 0.7, 0.2]), atol=1e-6)


def test_shadow_attack_learns_from_synthetic_shadows(separable_target_stats):
    """A shadow attack trained on separable shadow data must transfer.

    The shadow runs here share the target's structure (members confident,
    non-members not), so the classifier should pick it up and score well.
    """
    member_stats, nonmember_stats = separable_target_stats
    shadow_runs = [shadow_mia.ShadowRun(in_stats=member_stats, out_stats=nonmember_stats)]

    scores, labels, attack_metrics = shadow_mia.run_shadow_attack(
        shadow_runs, member_stats, nonmember_stats, classifier_kind="logistic_regression"
    )
    assert len(scores) == len(labels) == 400
    assert attack_metrics["auc"] > 0.9


def test_shadow_attack_requires_fit_before_scoring():
    attack = shadow_mia.ShadowAttack()
    with pytest.raises(RuntimeError, match="fit"):
        attack.predict_membership_scores(_make_stats(np.full((2, 10), 0.1), np.zeros(2, dtype=int)))


def test_unknown_classifier_kind_raises():
    with pytest.raises(ValueError, match="Unknown attack classifier"):
        shadow_mia._build_classifier("crystal_ball")


# ---------------- Phase 4: LiRA ----------------

def test_logit_scaled_confidence_hand_computed():
    """phi = logit_y - logsumexp(other logits), computed by hand.

    Logits [2, 1, 0] with true class 0:
      phi = 2 - log(e^1 + e^0) = 2 - log(3.71828...) = 2 - 1.31326 = 0.68674
    """
    stats = {
        "logits": np.array([[2.0, 1.0, 0.0]]),
        "true_label": np.array([0]),
    }
    phi = lira.logit_scaled_confidence(stats)
    expected = 2.0 - np.log(np.exp(1.0) + np.exp(0.0))
    assert phi[0] == pytest.approx(expected, abs=1e-9)


def test_logit_scaled_confidence_is_finite_at_saturation():
    """A perfectly memorized example (prob 1.0) must not produce inf.

    This is exactly why phi is computed from logits rather than probabilities.
    """
    stats = {
        "logits": np.array([[50.0, -50.0, -50.0]]),
        "true_label": np.array([0]),
    }
    phi = lira.logit_scaled_confidence(stats)
    assert np.isfinite(phi).all()
    assert phi[0] > 0


def test_logit_scaled_confidence_sign_tracks_correctness():
    # Confident and correct -> positive phi; confident and wrong -> negative.
    stats = {
        "logits": np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
        "true_label": np.array([0, 0]),
    }
    phi = lira.logit_scaled_confidence(stats)
    assert phi[0] > 0 > phi[1]


def test_lira_shadow_outputs_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must match"):
        lira.LiraShadowOutputs(phi=np.zeros((4, 10)), in_mask=np.zeros((4, 9), dtype=bool))


def test_lira_scores_rejects_misaligned_challenge_set():
    shadow_outputs = lira.LiraShadowOutputs(
        phi=np.zeros((4, 10)), in_mask=np.zeros((4, 10), dtype=bool)
    )
    with pytest.raises(ValueError, match="same challenge set"):
        lira.lira_scores(shadow_outputs, target_phi=np.zeros(5))


def test_lira_detects_membership_on_synthetic_shadows():
    """Construct shadows where being IN raises phi by a clear margin.

    8 shadows x 100 examples. Each example is IN for half the shadows, with
    phi ~ N(3, 1) when IN and N(0, 1) when OUT. The target's phi is drawn from
    the IN distribution for the first half of examples and OUT for the rest,
    so LiRA should separate them.
    """
    rng = np.random.default_rng(3)
    num_shadows, num_examples = 8, 100

    in_mask = np.zeros((num_shadows, num_examples), dtype=bool)
    in_mask[: num_shadows // 2, :] = True  # each example IN for exactly half

    phi = np.where(in_mask, rng.normal(3.0, 1.0, (num_shadows, num_examples)),
                   rng.normal(0.0, 1.0, (num_shadows, num_examples)))
    shadow_outputs = lira.LiraShadowOutputs(phi=phi, in_mask=in_mask)

    half = num_examples // 2
    target_phi = np.concatenate([rng.normal(3.0, 1.0, half), rng.normal(0.0, 1.0, half)])
    scores = lira.lira_scores(shadow_outputs, target_phi, variance_mode="global")

    labels = metrics.build_membership_labels(half, half)
    assert metrics.compute_auc(scores, labels) > 0.9


def test_lira_rejects_bad_variance_mode():
    shadow_outputs = lira.LiraShadowOutputs(
        phi=np.zeros((2, 3)), in_mask=np.ones((2, 3), dtype=bool)
    )
    with pytest.raises(ValueError, match="variance_mode"):
        lira.lira_scores(shadow_outputs, np.zeros(3), variance_mode="telekinesis")


def test_lira_handles_example_never_seen_in(caplog):
    """An example that is OUT for every shadow has no IN samples to fit.

    It must fall back to the global mean and warn, not crash or emit NaN.
    """
    phi = np.array([[1.0, 2.0], [1.5, 2.5]])
    in_mask = np.array([[True, False], [True, False]])  # column 1 is never IN
    shadow_outputs = lira.LiraShadowOutputs(phi=phi, in_mask=in_mask)

    with caplog.at_level("WARNING"):
        scores = lira.lira_scores(shadow_outputs, np.array([1.2, 2.2]))
    assert np.isfinite(scores).all()
    assert "no shadow samples" in caplog.text


# ---------------- config loader ----------------

def test_config_extends_merges_nested_sections(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "seed: 1\ndata:\n  a: 1\n  b: 2\ntrain:\n  epochs: 5\n"
    )
    (tmp_path / "child.yaml").write_text(
        "extends: base.yaml\nseed: 99\ndata:\n  b: 20\n"
    )
    config = utils.load_config(tmp_path / "child.yaml")

    assert config["seed"] == 99          # child overrides scalar
    assert config["data"] == {"a": 1, "b": 20}  # nested merge, not replacement
    assert config["train"] == {"epochs": 5}     # untouched section survives
    assert "extends" not in config
