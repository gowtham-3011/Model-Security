"""Evaluation metrics for membership inference and model extraction.

The central opinion baked into this module: **never judge a membership
inference attack by its accuracy alone.** See ``tpr_at_fpr`` for why.

Score convention used everywhere in this lab: a *higher* score means "more
likely to be a member". Attacks that naturally produce the opposite (loss,
where members score *low*) must negate before handing scores here.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)

# The low-FPR operating points we report everywhere. 0.1% is the headline.
DEFAULT_FPR_POINTS = (0.01, 0.001)


def build_membership_labels(n_members: int, n_nonmembers: int) -> np.ndarray:
    """Return the ground-truth label vector: 1 for members, 0 for non-members."""
    return np.concatenate([np.ones(n_members, dtype=int), np.zeros(n_nonmembers, dtype=int)])


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve.

    AUC is a useful *average-case* summary — it answers "for a random
    member/non-member pair, how often does the attack rank the member
    higher?" — but it is not sufficient on its own. An attack can post a
    respectable AUC of 0.6 purely by being slightly right about a large
    number of examples while never being *confident* about any of them, and
    such an attack is close to useless to a real adversary. Always read AUC
    alongside ``tpr_at_fpr``.
    """
    return float(roc_auc_score(labels, scores))


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    """True positive rate at a fixed, low false positive rate.

    **Why this is the key metric.** A membership inference attack is a real
    privacy threat when it can identify *some* training members with high
    confidence — not when it is vaguely better than chance on average.
    Carlini et al. (2022) made this point sharply: an attack with 60%
    balanced accuracy that is never confident about any individual example
    leaks far less than an attack that flags 1% of the training set with
    near-certainty, even though the latter may have a *worse* average-case
    score. Privacy is a worst-case property — the question that matters to
    the person whose medical record is in the training set is "can you
    single *me* out?", not "are you right 60% of the time overall?".

    So we evaluate in the regime a real attacker cares about: allow only a
    tiny false positive rate (1%, and especially 0.1%) and ask how many true
    members the attack still catches.

    We report the largest TPR achievable at an FPR *at or below* the target
    (the conservative reading of the empirical ROC step function). The
    alternative, linearly interpolating between ROC vertices, reports
    slightly higher numbers for operating points the attack cannot actually
    realize at this sample size, so we avoid it.
    """
    n_nonmembers = int((labels == 0).sum())
    min_resolvable_fpr = 1.0 / n_nonmembers if n_nonmembers else float("inf")
    if target_fpr < min_resolvable_fpr:
        logger.warning(
            "target_fpr=%.4g is below the resolution of this evaluation set: with %d "
            "non-members the smallest non-zero FPR is %.4g. Reported TPR will be 0 or "
            "meaningless — use more non-members.",
            target_fpr, n_nonmembers, min_resolvable_fpr,
        )

    fpr, tpr, _ = roc_curve(labels, scores)
    admissible = np.where(fpr <= target_fpr)[0]
    if admissible.size == 0:
        return 0.0
    return float(tpr[admissible[-1]])


def balanced_accuracy_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Balanced accuracy of the rule "predict member if score >= threshold".

    Balanced (rather than plain) accuracy because a member/non-member
    evaluation set need not be perfectly balanced, and plain accuracy on an
    unbalanced set can be beaten by a constant predictor.
    """
    predictions = (scores >= threshold).astype(int)
    members = labels == 1
    nonmembers = labels == 0
    tpr = float(predictions[members].mean()) if members.any() else 0.0
    tnr = float((1 - predictions[nonmembers]).mean()) if nonmembers.any() else 0.0
    return 0.5 * (tpr + tnr)


def roc_curve_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) arrays for plotting."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr


def attack_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
    fpr_points: tuple[float, ...] = DEFAULT_FPR_POINTS,
    with_ci: bool = False,
    n_resamples: int = 1000,
) -> dict[str, float]:
    """Compute the full metric bundle this lab reports for every MIA.

    ``threshold`` is only needed for balanced accuracy. AUC and TPR@FPR are
    threshold-free (they sweep every threshold), which is exactly why they
    are the primary metrics: they measure the attack's underlying signal
    rather than how well the attacker happened to calibrate one cut-off.

    ``with_ci=True`` adds bootstrap confidence intervals for AUC and the
    lowest FPR point. It's off by default because 1000 resamples of a
    low-FPR metric is not free, and defense sweeps call this hundreds of
    times; turn it on for headline comparison tables, where the reader needs
    to know which gaps are real.
    """
    metrics = {
        "auc": compute_auc(scores, labels),
        "n_members": int((labels == 1).sum()),
        "n_nonmembers": int((labels == 0).sum()),
    }
    for fpr_point in fpr_points:
        metrics[f"tpr_at_fpr_{fpr_point:g}"] = tpr_at_fpr(scores, labels, fpr_point)
    if threshold is not None:
        metrics["threshold"] = float(threshold)
        metrics["balanced_accuracy"] = balanced_accuracy_at_threshold(scores, labels, threshold)

    if with_ci:
        auc_low, auc_high = bootstrap_ci(scores, labels, "auc", n_resamples=n_resamples)
        metrics["auc_ci_low"], metrics["auc_ci_high"] = auc_low, auc_high

        # The strictest FPR point is where sampling noise bites hardest, so
        # that's the one whose interval the reader most needs.
        strictest_fpr = min(fpr_points)
        tpr_low, tpr_high = bootstrap_ci(
            scores, labels, "tpr_at_fpr", target_fpr=strictest_fpr, n_resamples=n_resamples
        )
        metrics[f"tpr_at_fpr_{strictest_fpr:g}_ci_low"] = tpr_low
        metrics[f"tpr_at_fpr_{strictest_fpr:g}_ci_high"] = tpr_high

    return metrics


def bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    metric_name: str = "auc",
    target_fpr: float = 0.001,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for an attack metric.

    Answers the question a point estimate can't: *is this difference real?*
    A table showing LiRA at 3.8% TPR and a baseline at 1.9% invites the
    reader to conclude LiRA is twice as strong — but with only a few thousand
    evaluation examples, 0.1%-FPR metrics rest on a handful of points and
    swing widely. Overlapping intervals mean "don't draw that conclusion yet".

    **Scope of this interval.** Resampling the evaluation set measures
    *evaluation* uncertainty — how much the number would move if you had
    drawn a different member/non-member sample from the same models. It does
    **not** capture training variance (a different seed producing a different
    target and different shadows), which is usually larger. For that you need
    multiple seeds; see ``summarize_over_seeds``. Quoting a bootstrap CI as
    though it covered run-to-run variation would understate the true
    uncertainty, so the two are reported separately and never merged.

    Members and non-members are resampled *separately*, preserving the
    original class balance in every resample. Pooling them would let the
    member/non-member ratio drift between resamples, which itself shifts FPR
    resolution and would widen the interval for a reason unrelated to the
    attack.
    """
    rng = np.random.default_rng(seed)
    member_indices = np.flatnonzero(labels == 1)
    nonmember_indices = np.flatnonzero(labels == 0)

    estimates = []
    for _ in range(n_resamples):
        resampled = np.concatenate(
            [
                rng.choice(member_indices, size=len(member_indices), replace=True),
                rng.choice(nonmember_indices, size=len(nonmember_indices), replace=True),
            ]
        )
        resampled_scores, resampled_labels = scores[resampled], labels[resampled]

        if metric_name == "auc":
            estimates.append(compute_auc(resampled_scores, resampled_labels))
        elif metric_name == "tpr_at_fpr":
            estimates.append(tpr_at_fpr(resampled_scores, resampled_labels, target_fpr))
        else:
            raise ValueError(f"metric_name must be 'auc' or 'tpr_at_fpr', got '{metric_name}'.")

    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def fidelity(surrogate_predictions: np.ndarray, target_predictions: np.ndarray) -> float:
    """Agreement between a stolen surrogate and its target: the fraction of
    inputs on which both predict the same label.

    This is the metric that matters for model *extraction*, as distinct from
    the surrogate's task accuracy. A surrogate can be accurate on CIFAR-10
    while disagreeing with the target often (it learned the task, not the
    model); high fidelity means it replicated the target's decision
    boundary, including its mistakes — which is what makes it a stepping
    stone for further attacks.
    """
    return float((surrogate_predictions == target_predictions).mean())


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Plain classification accuracy."""
    return float((predictions == labels).mean())


def summarize_over_seeds(runs: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate a list of per-seed metric dicts into mean/std per key.

    Section 5 of the lab spec requires >=3 seeds with mean +/- std wherever
    feasible: single-seed attack numbers move by several points run to run,
    so an unaggregated comparison between two defenses can easily be pure
    noise.
    """
    if not runs:
        return {}
    summary: dict[str, float] = {"n_seeds": len(runs)}
    for key in runs[0]:
        values = np.array([run[key] for run in runs], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary
