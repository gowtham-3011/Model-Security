"""Phase 2: the loss/confidence-threshold membership inference baseline.

The attack in one sentence: models fit their training data better than
unseen data, so an example with unusually *low* loss is probably a member.

This is the weakest attack in the lab and that's the point — it establishes
the floor that Phase 3 (shadow models) and Phase 4 (LiRA) must beat, and it
is the attack that a generalization gap alone is enough to enable.

Threshold calibration, and why it's subtle: the attacker does not know which
examples are members (that's what they're trying to learn), so they cannot
tune a threshold on the target's own data. Instead they train a *shadow*
model on their own auxiliary data, where they *do* know the membership
labels by construction, pick the threshold that works best there, and
transfer it to the target. This is Yeom et al.'s calibration strategy and it
keeps the attacker strictly honest: ``attacker_aux`` never overlaps the
target's member/non-member evaluation sets (guaranteed by src.data).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .. import metrics

logger = logging.getLogger(__name__)


def loss_scores(stats: dict[str, np.ndarray]) -> np.ndarray:
    """Membership score from per-example loss.

    Negated because this lab's convention is "higher score = more likely
    member", while members have *lower* loss.
    """
    return -stats["loss"]


def confidence_scores(stats: dict[str, np.ndarray]) -> np.ndarray:
    """Membership score from the model's confidence in the *true* label.

    Already oriented correctly: members tend to get higher true-label
    confidence. Note this uses the true label, so it assumes an attacker who
    knows the label of the example they're testing — the standard MIA threat
    model (they're testing "was this labeled record in your training set?").

    **This score is rank-equivalent to ``loss_scores``, by construction.**
    Cross-entropy loss *is* ``-log(confidence_true_label)``, so
    ``-loss == log(confidence)`` — a strictly increasing transform of
    confidence. Any metric that depends only on the *ordering* of scores
    (AUC, TPR@FPR, the full ROC) is therefore mathematically identical for
    the two, and empirical runs confirm it to the last decimal place.

    They are both kept because they differ in the one thing ordering doesn't
    determine: the calibrated threshold, and hence balanced accuracy — a
    threshold transfers between models only as well as the score's *scale*
    transfers, and log-probabilities and probabilities behave differently
    there. Treat them as one attack with two calibrations, not two attacks;
    a comparison table that lists them as separate rows with identical AUC
    is reporting one result twice.
    """
    return stats["confidence_true_label"]


SCORE_FUNCTIONS = {
    "loss": loss_scores,
    "confidence": confidence_scores,
}


def choose_threshold(
    aux_member_scores: np.ndarray,
    aux_nonmember_scores: np.ndarray,
) -> float:
    """Pick the threshold maximizing balanced accuracy on the attacker's
    shadow-model scores.

    We sweep every midpoint between observed score values rather than
    assuming a parametric form, because the score distributions are strongly
    non-Gaussian (loss piles up near zero for a memorized training set).
    """
    scores = np.concatenate([aux_member_scores, aux_nonmember_scores])
    labels = metrics.build_membership_labels(len(aux_member_scores), len(aux_nonmember_scores))

    candidates = np.unique(scores)
    if candidates.size > 1:
        # Midpoints between adjacent observed values: the only places where
        # the decision rule's behaviour actually changes.
        candidates = (candidates[:-1] + candidates[1:]) / 2.0

    best_threshold, best_score = float(candidates[0]), -1.0
    for candidate in candidates:
        balanced = metrics.balanced_accuracy_at_threshold(scores, labels, candidate)
        if balanced > best_score:
            best_threshold, best_score = float(candidate), balanced

    logger.info(
        "Calibrated threshold=%.6f on attacker auxiliary data (shadow balanced acc=%.4f)",
        best_threshold, best_score,
    )
    return best_threshold


@dataclass
class ThresholdAttackResult:
    """Outcome of one threshold attack, ready to be logged or plotted."""

    score_name: str
    scores: np.ndarray
    labels: np.ndarray
    threshold: float
    metrics: dict[str, float] = field(default_factory=dict)


def run_threshold_attack(
    member_stats: dict[str, np.ndarray],
    nonmember_stats: dict[str, np.ndarray],
    score_name: str = "loss",
    threshold: float | None = None,
    with_ci: bool = False,
) -> ThresholdAttackResult:
    """Run the threshold attack against a target's member/non-member stats.

    ``threshold`` should come from ``choose_threshold`` on auxiliary data. If
    omitted, balanced accuracy is skipped but AUC and TPR@FPR are still
    reported — those two sweep all thresholds, so they measure the signal
    available to the attacker independently of calibration quality. That
    distinction is worth internalizing: a badly calibrated threshold hurts
    balanced accuracy but cannot hide leakage from AUC/TPR@FPR.
    """
    if score_name not in SCORE_FUNCTIONS:
        raise ValueError(f"Unknown score_name '{score_name}'. Available: {sorted(SCORE_FUNCTIONS)}")
    score_fn = SCORE_FUNCTIONS[score_name]

    member_scores = score_fn(member_stats)
    nonmember_scores = score_fn(nonmember_stats)
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = metrics.build_membership_labels(len(member_scores), len(nonmember_scores))

    return ThresholdAttackResult(
        score_name=score_name,
        scores=scores,
        labels=labels,
        threshold=float("nan") if threshold is None else threshold,
        metrics=metrics.attack_metrics(scores, labels, threshold=threshold, with_ci=with_ci),
    )
