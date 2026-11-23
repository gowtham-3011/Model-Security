"""Phase 6: defenses — output hardening and DP-SGD.

Three families of defense live in this lab, and they differ in *what* they
protect, which is the most important thing to understand here:

1. **Regularization** (weight decay, dropout, early stopping, label
   smoothing). No code needed — these are training hyperparameters, set in
   YAML and consumed by ``train.train_model``. They reduce the
   generalization gap, which reduces the *average-case* leakage the Phase 2
   attack depends on. They do **not** come with a guarantee, and they barely
   dent per-example memorization, so expect them to move AUC a lot more than
   they move TPR@0.1%FPR.

2. **Output hardening** (this module). Truncating or coarsening what the API
   returns. Cheap, no retraining, and effective against *model extraction*,
   which feeds on the information content of each response. Against
   membership inference it is much weaker than it looks: returning only the
   top-1 label still leaks whether the model got the example right, and on an
   overfit model that alone separates members from non-members.

3. **DP-SGD** (this module). The only defense here with a formal guarantee.
   It bounds how much any single training example can influence the final
   weights, which is exactly the quantity membership inference exploits.
   It costs accuracy, and that trade-off is the lab's headline result.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def harden_probabilities(
    probs: torch.Tensor,
    mode: str = "full",
    top_k: int | None = None,
    round_digits: int | None = None,
) -> torch.Tensor:
    """Apply output hardening to a batch of probability vectors.

    Args:
        probs: (batch, num_classes) probabilities that sum to 1 per row.
        mode: ``"full"`` returns the vector as-is; ``"label"`` returns a
            one-hot of the argmax — the strongest hardening, and the setting
            a cautious production API would use.
        top_k: keep only the k largest probabilities, zero the rest, and
            renormalize so rows still sum to 1.
        round_digits: round probabilities to this many decimals, then
            renormalize. Coarsening the *precision* of confidences is what
            kills the fine-grained signal extraction attacks rely on.

    ``top_k`` and ``round_digits`` compose, and both are ignored under
    ``mode="label"`` (a one-hot vector has nothing left to truncate).

    Renormalization after truncation/rounding is deliberate: an API that
    returned rows summing to 0.83 would itself leak how much mass was
    removed, handing the attacker back some of the information we just took
    away.
    """
    if mode not in ("full", "label"):
        raise ValueError(f"mode must be 'full' or 'label', got '{mode}'.")

    if mode == "label":
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, probs.argmax(dim=1, keepdim=True), 1.0)
        return one_hot

    hardened = probs.clone()

    if top_k is not None:
        if not 1 <= top_k <= probs.shape[1]:
            raise ValueError(f"top_k must be in [1, {probs.shape[1]}], got {top_k}.")
        kept_values, kept_indices = hardened.topk(top_k, dim=1)
        hardened = torch.zeros_like(hardened).scatter_(1, kept_indices, kept_values)

    if round_digits is not None:
        if round_digits < 0:
            raise ValueError(f"round_digits must be non-negative, got {round_digits}.")
        scale = 10.0**round_digits
        hardened = torch.round(hardened * scale) / scale

    row_sums = hardened.sum(dim=1, keepdim=True)
    # A row can round to all-zeros (e.g. round_digits=0 on a flat distribution);
    # fall back to the un-hardened row rather than emitting NaNs.
    degenerate = row_sums.squeeze(1) <= 0
    if degenerate.any():
        hardened[degenerate] = probs[degenerate]
        row_sums = hardened.sum(dim=1, keepdim=True)

    return hardened / row_sums


def apply_hardening_to_stats(
    stats: dict[str, np.ndarray],
    mode: str = "full",
    top_k: int | None = None,
    round_digits: int | None = None,
) -> dict[str, np.ndarray]:
    """Recompute per-example attack statistics as seen through a hardened API.

    Output hardening is usually pitched as an anti-extraction defense, but it
    changes what a membership inference attacker can compute too, so we must
    re-derive their features rather than scoring them against the raw outputs
    the defender never actually returns.

    The important case is ``mode="label"``: with only a top-1 label, the
    attacker can no longer compute a real loss. The best they can do is ask
    "did the model get this example right?", which is Yeom et al.'s
    correctness baseline — and on an overfit model that single bit is still a
    usable membership signal. Modelling that faithfully is why this function
    exists: scoring label-only APIs against un-hardened losses would make the
    defense look far stronger than it is.
    """
    probs = torch.from_numpy(stats["probs"].astype(np.float32))
    hardened = harden_probabilities(probs, mode=mode, top_k=top_k, round_digits=round_digits).numpy()

    true_labels = stats["true_label"].astype(int)
    row_indices = np.arange(len(true_labels))
    true_class_probs = hardened[row_indices, true_labels]

    # Clamp before the log: hardening drives most entries to exactly 0, and
    # log(0) would be -inf, which breaks AUC computation.
    epsilon = 1e-12
    hardened_stats = dict(stats)
    hardened_stats["probs"] = hardened
    hardened_stats["confidence_true_label"] = true_class_probs
    hardened_stats["loss"] = -np.log(np.clip(true_class_probs, epsilon, 1.0))
    hardened_stats["max_prob"] = hardened.max(axis=1)
    hardened_stats["predicted_label"] = hardened.argmax(axis=1)
    return hardened_stats


def validate_dp_compatible(model: nn.Module) -> list[str]:
    """Return Opacus's list of complaints about ``model``, empty if it's clean.

    Called before DP training so an incompatible layer produces a clear error
    up front instead of a confusing failure deep inside the privacy engine.
    Our models use GroupNorm precisely to keep this list empty — see
    src/models.py.
    """
    from opacus.validators import ModuleValidator

    return [str(error) for error in ModuleValidator.validate(model, strict=False)]


def make_private(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    epochs: int,
    target_epsilon: float,
    target_delta: float,
    max_grad_norm: float = 1.0,
) -> tuple[nn.Module, torch.optim.Optimizer, DataLoader, Any]:
    """Wrap a model/optimizer/loader for DP-SGD training with Opacus.

    Returns ``(model, optimizer, data_loader, privacy_engine)``, all replaced
    by their private counterparts.

    Design notes on the DP settings, since these are the easiest things in the
    whole lab to get subtly wrong:

    * **We target epsilon, not a noise multiplier.** ``make_private_with_epsilon``
      inverts the RDP accountant to find the noise level that lands exactly on
      the requested epsilon after ``epochs`` of training at this sampling
      rate. Specifying noise directly instead would mean the actual epsilon
      depended on epochs and batch size, making cross-run comparison
      meaningless — which is the whole point of the epsilon sweep.

    * **max_grad_norm (C) is the clipping bound**, applied to each *per-sample*
      gradient before averaging. This is what bounds one example's influence
      and makes the noise calibration valid: without clipping there is no
      finite sensitivity to calibrate against. C=1.0 is the usual starting
      point; too small and the signal is destroyed, too large and the required
      noise grows with it.

    * **delta should be <= 1/N** (the caller passes 1/N by convention). delta is
      the probability the guarantee simply fails, so it must be small relative
      to the dataset — a delta of 1/100 on 10,000 records would permit
      leaking ~100 of them outright.

    * **Opacus replaces the DataLoader** with one that does Poisson sampling
      (each example included independently with probability B/N) because the
      privacy accounting assumes exactly that, not the fixed-size shuffled
      batches a normal DataLoader produces. This is why the returned loader
      must be used instead of the one passed in.
    """
    from opacus import PrivacyEngine

    complaints = validate_dp_compatible(model)
    if complaints:
        raise ValueError(
            "Model is not compatible with Opacus DP-SGD:\n  - "
            + "\n  - ".join(complaints)
            + "\nUse GroupNorm instead of BatchNorm (see src/models.py), or run "
            "opacus.validators.ModuleValidator.fix(model) first."
        )

    privacy_engine = PrivacyEngine()
    private_model, private_optimizer, private_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )
    logger.info(
        "DP-SGD engaged: target_epsilon=%.4g target_delta=%.3g max_grad_norm=%.2f "
        "-> noise_multiplier=%.4f",
        target_epsilon, target_delta, max_grad_norm, private_optimizer.noise_multiplier,
    )
    return private_model, private_optimizer, private_loader, privacy_engine


def delta_for_dataset_size(n_examples: int) -> float:
    """The lab's delta convention: 1/N, rounded to a clean power of ten.

    Fixing delta this way (rather than per-experiment tuning) keeps the
    epsilon sweep an apples-to-apples comparison.
    """
    if n_examples <= 0:
        raise ValueError(f"n_examples must be positive, got {n_examples}.")
    return 1.0 / n_examples
