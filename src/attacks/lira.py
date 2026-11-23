"""Phase 4: LiRA — the Likelihood Ratio Attack (Carlini et al., 2022).

**Honest scope statement first.** This is a *simplified* LiRA. The paper
trains 256 shadow models; this lab trains 8-32 because it has to run on a
free Colab GPU. It also omits the paper's query augmentation (averaging the
statistic over flips/crops of each example). Both shortcuts cost real attack
power, so the TPR@0.1%FPR reported here should be read as a **lower bound**
on what LiRA achieves, not a reproduction of the paper's numbers.

## Why LiRA is stronger than everything before it

The Phase 2 and Phase 3 attacks both ask a *global* question: "is this loss
low enough to look like a member?" That question is unfair to hard examples
and far too lenient on easy ones. A blurry, atypical image might have high
loss even when memorized; a canonical, easy image has near-zero loss whether
or not it was trained on. A global threshold therefore spends its false
positives on easy non-members and misses memorized hard ones.

LiRA asks a *per-example* question instead: for **this specific example**,
what does the confidence distribution look like across models that trained
on it, versus models that didn't? Those two distributions are estimated
empirically from shadow models, and the attack is the likelihood ratio
between them:

    Lambda(x) = log p(phi_target(x) | IN) - log p(phi_target(x) | OUT)

Calibrating per example is precisely what buys the high TPR at low FPR: an
easy example whose confidence is high under *both* hypotheses contributes no
evidence either way, so the attack stops wasting false positives on it.

## The logit-scaling step

Softmax confidence on the true class is bounded in [0,1] and piles up against
1.0 for memorized examples — badly non-Gaussian, so fitting a Gaussian to it
directly is invalid. LiRA first applies a logit transform,

    phi = log( p_y / (1 - p_y) )

which maps (0,1) to the whole real line and empirically produces something
close enough to Gaussian for the parametric likelihood ratio to work. We
compute it from raw logits via logsumexp rather than from probabilities,
because p_y saturates to exactly 1.0 in float32 on memorized examples and the
naive formula would divide by zero.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import norm
from torch.utils.data import DataLoader, Dataset

from .. import data as data_module
from .. import metrics, models, train, utils

logger = logging.getLogger(__name__)

# Floor on fitted standard deviations. With few shadow models a per-example
# variance estimate can collapse toward 0, which would make the Gaussian a
# spike and the log-likelihood explode. Carlini et al. apply the same kind of
# stabilization.
MIN_STD = 1e-3


def logit_scaled_confidence(stats: dict[str, np.ndarray]) -> np.ndarray:
    """Compute phi = log(p_y / (1 - p_y)) stably from raw logits.

    Using the identity  log(p_y) - log(1 - p_y)
                      = logit_y - logsumexp(all logits)
                        - [ logsumexp(logits excluding y) - logsumexp(all logits) ]
                      = logit_y - logsumexp(logits excluding y)
    which never forms the quantity (1 - p_y) explicitly and so never
    underflows to zero for a confidently-memorized example.
    """
    logits = stats["logits"].astype(np.float64)
    true_labels = stats["true_label"].astype(int)

    row_indices = np.arange(len(true_labels))
    true_class_logit = logits[row_indices, true_labels]

    other_logits = logits.copy()
    other_logits[row_indices, true_labels] = -np.inf  # exclude the true class
    return true_class_logit - logsumexp(other_logits, axis=1)


@dataclass
class LiraShadowOutputs:
    """Per-example shadow statistics aligned to a fixed challenge set.

    ``phi`` and ``in_mask`` are both (num_shadows, num_challenge_examples),
    column *i* of both referring to the same challenge example.
    """

    phi: np.ndarray
    in_mask: np.ndarray

    def __post_init__(self) -> None:
        if self.phi.shape != self.in_mask.shape:
            raise ValueError(
                f"phi shape {self.phi.shape} must match in_mask shape {self.in_mask.shape}."
            )


def train_lira_shadows(
    pool: Dataset,
    challenge_indices: np.ndarray,
    num_shadows: int,
    model_config: dict,
    train_config: dict,
    device: torch.device,
    in_fraction: float = 0.5,
    seed: int = 0,
) -> LiraShadowOutputs:
    """Train shadow models on random halves of the challenge set (online LiRA).

    ``in_fraction=0.5`` is the key design choice: each example ends up IN for
    roughly half the shadows, which maximizes the number of samples available
    for *both* of its per-example Gaussians. This is the "online" setting —
    the attacker is allowed to train shadows on the very points they will
    later test. The weaker "offline" variant (OUT-only shadows, for an
    attacker who cannot include challenge points) is not implemented here.

    Each shadow's statistics are computed over the *entire* challenge set, so
    the returned matrices are column-aligned regardless of which examples that
    shadow trained on.
    """
    in_size = int(round(in_fraction * len(challenge_indices)))
    challenge_subset = data_module.get_subset(pool, challenge_indices)

    phi_rows, mask_rows = [], []
    for shadow_index in range(num_shadows):
        shadow_seed = seed + 5000 + shadow_index
        utils.set_seed(shadow_seed)

        in_indices, out_indices = data_module.make_in_out_split(
            challenge_indices, in_size, seed=shadow_seed
        )
        model = models.build_model(**model_config)
        logger.info("Training LiRA shadow %d/%d (in=%d)", shadow_index + 1, num_shadows, in_size)

        model, _ = train.train_model(
            model,
            DataLoader(data_module.get_subset(pool, in_indices),
                       batch_size=train_config["batch_size"], shuffle=True),
            DataLoader(data_module.get_subset(pool, out_indices),
                       batch_size=train_config["batch_size"], shuffle=False),
            device,
            epochs=train_config["epochs"],
            lr=train_config["lr"],
            weight_decay=train_config.get("weight_decay", 0.0),
            label_smoothing=train_config.get("label_smoothing", 0.0),
            early_stopping_patience=train_config.get("early_stopping_patience"),
        )

        stats = train.compute_per_example_stats(model, challenge_subset, device)
        phi_rows.append(logit_scaled_confidence(stats))
        mask_rows.append(np.isin(challenge_indices, in_indices))

    return LiraShadowOutputs(phi=np.vstack(phi_rows), in_mask=np.vstack(mask_rows))


def _fit_gaussians(
    shadow_outputs: LiraShadowOutputs,
    variance_mode: str = "global",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit per-example IN and OUT Gaussians from shadow statistics.

    Returns (mu_in, sigma_in, mu_out, sigma_out), each length num_challenge.

    ``variance_mode="global"`` shares one pooled standard deviation across all
    examples (per hypothesis). This is the right default at this lab's shadow
    counts: with 16 shadows an example has only ~8 IN samples, and a
    per-example variance from 8 points is so noisy that it *hurts* the
    likelihood ratio more than the extra flexibility helps. Use
    ``"per_example"`` only with many shadows.
    """
    if variance_mode not in ("global", "per_example"):
        raise ValueError(f"variance_mode must be 'global' or 'per_example', got '{variance_mode}'.")

    phi, in_mask = shadow_outputs.phi, shadow_outputs.in_mask
    out_mask = ~in_mask

    # Masked means: examples never seen IN (or never OUT) get NaN here and are
    # patched from the global mean below.
    with np.errstate(invalid="ignore"):
        mu_in = np.where(in_mask.sum(axis=0) > 0, np.nansum(np.where(in_mask, phi, np.nan), axis=0)
                         / np.maximum(in_mask.sum(axis=0), 1), np.nan)
        mu_out = np.where(out_mask.sum(axis=0) > 0, np.nansum(np.where(out_mask, phi, np.nan), axis=0)
                          / np.maximum(out_mask.sum(axis=0), 1), np.nan)

    global_mu_in = np.nanmean(mu_in) if np.isfinite(mu_in).any() else 0.0
    global_mu_out = np.nanmean(mu_out) if np.isfinite(mu_out).any() else 0.0
    n_missing = int(np.isnan(mu_in).sum() + np.isnan(mu_out).sum())
    if n_missing:
        logger.warning(
            "%d example-hypothesis pairs had no shadow samples; falling back to the "
            "global mean for them. Increase num_shadows to avoid this.", n_missing,
        )
    mu_in = np.nan_to_num(mu_in, nan=global_mu_in)
    mu_out = np.nan_to_num(mu_out, nan=global_mu_out)

    # Residuals about each example's own mean, under each hypothesis.
    residuals_in = np.where(in_mask, phi - mu_in, np.nan)
    residuals_out = np.where(out_mask, phi - mu_out, np.nan)

    if variance_mode == "global":
        sigma_in_scalar = float(np.nanstd(residuals_in)) if np.isfinite(residuals_in).any() else MIN_STD
        sigma_out_scalar = float(np.nanstd(residuals_out)) if np.isfinite(residuals_out).any() else MIN_STD
        sigma_in = np.full_like(mu_in, max(sigma_in_scalar, MIN_STD))
        sigma_out = np.full_like(mu_out, max(sigma_out_scalar, MIN_STD))
    else:
        # An example with 0 or 1 samples under a hypothesis has no usable
        # variance; numpy warns and returns NaN, which we expect and patch to
        # MIN_STD below. Suppressed so the warning doesn't drown real ones.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sigma_in = np.nanstd(residuals_in, axis=0)
            sigma_out = np.nanstd(residuals_out, axis=0)
        sigma_in = np.maximum(np.nan_to_num(sigma_in, nan=MIN_STD), MIN_STD)
        sigma_out = np.maximum(np.nan_to_num(sigma_out, nan=MIN_STD), MIN_STD)

    return mu_in, sigma_in, mu_out, sigma_out


def lira_scores(
    shadow_outputs: LiraShadowOutputs,
    target_phi: np.ndarray,
    variance_mode: str = "global",
) -> np.ndarray:
    """Compute the LiRA log-likelihood ratio for each challenge example.

    Higher = more member-like, matching this lab's score convention.
    """
    if len(target_phi) != shadow_outputs.phi.shape[1]:
        raise ValueError(
            f"target_phi has {len(target_phi)} examples but shadow statistics cover "
            f"{shadow_outputs.phi.shape[1]}. They must describe the same challenge set, in order."
        )

    mu_in, sigma_in, mu_out, sigma_out = _fit_gaussians(shadow_outputs, variance_mode)
    log_p_in = norm.logpdf(target_phi, loc=mu_in, scale=sigma_in)
    log_p_out = norm.logpdf(target_phi, loc=mu_out, scale=sigma_out)
    return log_p_in - log_p_out


def run_lira_attack(
    shadow_outputs: LiraShadowOutputs,
    target_member_stats: dict[str, np.ndarray],
    target_nonmember_stats: dict[str, np.ndarray],
    variance_mode: str = "global",
    with_ci: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Score the target's members and non-members with LiRA.

    The challenge set must be ordered members-then-non-members, matching how
    ``train_lira_shadows`` was called, so that shadow columns line up with the
    target statistics concatenated here.
    """
    target_phi = np.concatenate(
        [logit_scaled_confidence(target_member_stats),
         logit_scaled_confidence(target_nonmember_stats)]
    )
    scores = lira_scores(shadow_outputs, target_phi, variance_mode=variance_mode)
    labels = metrics.build_membership_labels(
        len(target_member_stats["loss"]), len(target_nonmember_stats["loss"])
    )
    # Threshold 0 is the natural cut for a log-likelihood ratio: positive means
    # the IN hypothesis is more likely than OUT. It needs no target data to set.
    return scores, labels, metrics.attack_metrics(
        scores, labels, threshold=0.0, with_ci=with_ci
    )
