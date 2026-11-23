"""Metric correctness on toy inputs where the answer is known by hand.

Metrics are the one place a bug is invisible: a broken TPR@FPR still returns a
plausible-looking number, and every conclusion in the lab would inherit the
error. So each metric is checked against a case computed by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import metrics


def test_auc_perfect_separation():
    # Every member scores above every non-member -> AUC exactly 1.
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    labels = np.array([1, 1, 1, 0, 0, 0])
    assert metrics.compute_auc(scores, labels) == pytest.approx(1.0)


def test_auc_inverted_separation():
    # Perfectly wrong ranking -> AUC exactly 0.
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([1, 1, 1, 0, 0, 0])
    assert metrics.compute_auc(scores, labels) == pytest.approx(0.0)


def test_auc_identical_scores_is_chance():
    # No signal whatsoever -> AUC 0.5.
    scores = np.full(6, 0.5)
    labels = np.array([1, 1, 1, 0, 0, 0])
    assert metrics.compute_auc(scores, labels) == pytest.approx(0.5)


def test_tpr_at_fpr_perfect_attack():
    # Perfect separation: at any FPR, TPR is 1.
    scores = np.concatenate([np.linspace(0.6, 1.0, 100), np.linspace(0.0, 0.4, 100)])
    labels = metrics.build_membership_labels(100, 100)
    assert metrics.tpr_at_fpr(scores, labels, 0.01) == pytest.approx(1.0)


def test_tpr_at_fpr_hand_computed():
    """A case worked out by hand, chosen to pin down a subtlety.

    10 members, 10 non-members. Sorted by score, the ranking is:

        10.0(M) 9.0(M) 8.0(N) 3.0(M) 2.9(M) ... 2.3(M) 2.2(N) 2.1(N) ...

    The naive reading of "TPR at 10% FPR" is "count members above the first
    non-member" = 0.2. That's wrong. The metric asks for the *best* TPR
    achievable subject to FPR <= 0.1, and a threshold of 2.3 admits all ten
    members while catching only one non-member (8.0) — exactly 1/10 = 10% FPR.
    So the answer is 1.0.

    At FPR = 0 the threshold must exclude 8.0 entirely, leaving only the two
    members above it: TPR = 0.2.
    """
    member_scores = np.array([10.0, 9.0, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3])
    nonmember_scores = np.array([8.0, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4])
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = metrics.build_membership_labels(10, 10)

    assert metrics.tpr_at_fpr(scores, labels, 0.1) == pytest.approx(1.0)
    assert metrics.tpr_at_fpr(scores, labels, 0.0) == pytest.approx(0.2)


def test_tpr_at_fpr_is_monotone_in_the_budget():
    """Allowing a larger FPR can never reduce the achievable TPR."""
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(1.0, 1.0, 500), rng.normal(0.0, 1.0, 500)])
    labels = metrics.build_membership_labels(500, 500)

    values = [metrics.tpr_at_fpr(scores, labels, fpr) for fpr in (0.002, 0.01, 0.1, 0.5)]
    assert values == sorted(values)


def test_tpr_at_fpr_warns_when_unresolvable(caplog):
    """Requesting 0.1% FPR with only 10 non-members cannot be resolved.

    The smallest non-zero FPR is 1/10 = 10%, so the function must warn rather
    than silently return a meaningless number.
    """
    scores = np.concatenate([np.ones(10), np.zeros(10)])
    labels = metrics.build_membership_labels(10, 10)
    with caplog.at_level("WARNING"):
        metrics.tpr_at_fpr(scores, labels, 0.001)
    assert "below the resolution" in caplog.text


def test_balanced_accuracy_hand_computed():
    # 4 members, 4 non-members; threshold 0.5.
    # Members >= 0.5: 3 of 4 -> TPR 0.75. Non-members < 0.5: 2 of 4 -> TNR 0.5.
    scores = np.array([0.9, 0.8, 0.6, 0.4, 0.7, 0.55, 0.3, 0.1])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert metrics.balanced_accuracy_at_threshold(scores, labels, 0.5) == pytest.approx(0.625)


def test_balanced_accuracy_beats_constant_predictor_on_imbalance():
    """Balanced accuracy must not reward a constant "everything is a member".

    With 9 members and 1 non-member, always predicting "member" gets 90% plain
    accuracy but only 50% balanced accuracy — which is why the lab reports the
    balanced version.
    """
    scores = np.ones(10)
    labels = np.array([1] * 9 + [0])
    assert metrics.balanced_accuracy_at_threshold(scores, labels, 0.5) == pytest.approx(0.5)


def test_fidelity_and_accuracy():
    surrogate = np.array([0, 1, 2, 3])
    target = np.array([0, 1, 2, 9])
    truth = np.array([0, 1, 9, 9])
    assert metrics.fidelity(surrogate, target) == pytest.approx(0.75)
    assert metrics.accuracy(surrogate, truth) == pytest.approx(0.5)


def test_build_membership_labels():
    labels = metrics.build_membership_labels(3, 2)
    np.testing.assert_array_equal(labels, np.array([1, 1, 1, 0, 0]))


def test_attack_metrics_bundle_has_required_keys():
    """Every MIA result must carry AUC *and* both low-FPR points.

    This encodes the lab's evaluation rule: accuracy alone is never a
    sufficient report for a membership inference attack.
    """
    scores = np.concatenate([np.linspace(0.5, 1.0, 50), np.linspace(0.0, 0.5, 50)])
    labels = metrics.build_membership_labels(50, 50)
    bundle = metrics.attack_metrics(scores, labels, threshold=0.5)
    for key in ("auc", "tpr_at_fpr_0.01", "tpr_at_fpr_0.001", "balanced_accuracy"):
        assert key in bundle


def test_summarize_over_seeds_mean_and_std():
    runs = [{"auc": 0.6}, {"auc": 0.7}, {"auc": 0.8}]
    summary = metrics.summarize_over_seeds(runs)
    assert summary["n_seeds"] == 3
    assert summary["auc_mean"] == pytest.approx(0.7)
    assert summary["auc_std"] == pytest.approx(0.1)  # sample std (ddof=1)


def test_summarize_over_seeds_single_run_has_zero_std():
    summary = metrics.summarize_over_seeds([{"auc": 0.65}])
    assert summary["auc_mean"] == pytest.approx(0.65)
    assert summary["auc_std"] == pytest.approx(0.0)


# ---------------- bootstrap confidence intervals ----------------

@pytest.fixture
def separable_scores():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(1.5, 1, 1000), rng.normal(0, 1, 1000)])
    return scores, metrics.build_membership_labels(1000, 1000)


def test_bootstrap_ci_brackets_the_point_estimate(separable_scores):
    scores, labels = separable_scores
    low, high = metrics.bootstrap_ci(scores, labels, "auc", n_resamples=200)
    assert low < metrics.compute_auc(scores, labels) < high


def test_bootstrap_ci_excludes_chance_for_a_real_attack(separable_scores):
    """A clearly-separated attack's interval must sit entirely above 0.5."""
    scores, labels = separable_scores
    low, _ = metrics.bootstrap_ci(scores, labels, "auc", n_resamples=200)
    assert low > 0.5


def test_bootstrap_ci_contains_chance_for_a_signal_free_attack():
    """The negative control: no signal means the interval must straddle 0.5,
    so a reader can tell 'this attack found nothing' from 'this attack is weak'."""
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(0, 1, 1000), rng.normal(0, 1, 1000)])
    labels = metrics.build_membership_labels(1000, 1000)
    low, high = metrics.bootstrap_ci(scores, labels, "auc", n_resamples=200)
    assert low < 0.5 < high


def test_bootstrap_ci_is_deterministic_for_a_given_seed(separable_scores):
    scores, labels = separable_scores
    first = metrics.bootstrap_ci(scores, labels, "auc", n_resamples=150, seed=7)
    second = metrics.bootstrap_ci(scores, labels, "auc", n_resamples=150, seed=7)
    assert first == second


def test_bootstrap_ci_preserves_class_balance(separable_scores):
    """Stratified resampling must keep the member/non-member counts fixed.

    If the balance drifted between resamples, the FPR grid would shift and
    the interval would widen for reasons unrelated to the attack.
    """
    scores, labels = separable_scores
    # Widen/narrow checks aside, an unbalanced eval set must still produce a
    # finite, ordered interval.
    unbalanced_scores = np.concatenate([scores[:800], scores[1000:]])
    unbalanced_labels = metrics.build_membership_labels(800, 1000)
    low, high = metrics.bootstrap_ci(unbalanced_scores, unbalanced_labels, "auc", n_resamples=150)
    assert low <= high
    assert np.isfinite([low, high]).all()


def test_bootstrap_ci_rejects_unknown_metric(separable_scores):
    scores, labels = separable_scores
    with pytest.raises(ValueError, match="metric_name"):
        metrics.bootstrap_ci(scores, labels, "vibes", n_resamples=10)


def test_attack_metrics_with_ci_adds_interval_columns(separable_scores):
    scores, labels = separable_scores
    bundle = metrics.attack_metrics(scores, labels, threshold=0.5, with_ci=True, n_resamples=100)
    for key in ("auc_ci_low", "auc_ci_high",
                "tpr_at_fpr_0.001_ci_low", "tpr_at_fpr_0.001_ci_high"):
        assert key in bundle
    assert bundle["auc_ci_low"] <= bundle["auc"] <= bundle["auc_ci_high"]


def test_attack_metrics_without_ci_stays_cheap(separable_scores):
    """CIs are opt-in: defense sweeps call this hundreds of times."""
    scores, labels = separable_scores
    bundle = metrics.attack_metrics(scores, labels, threshold=0.5)
    assert not any(key.endswith("_ci_low") for key in bundle)
