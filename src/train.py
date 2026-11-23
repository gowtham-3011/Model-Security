"""Standard training loop, plus per-example statistic extraction that every
later MIA attack (Phase 2+) is built on.

DP-SGD training (Opacus) is added to this same file in Phase 6, once
defenses.py exists to configure it — keeping both training modes in one
file (as the repo layout calls for) means attacks and evaluation code don't
need to care which one produced a given checkpoint.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """Run one pass over ``loader``. Trains if ``optimizer`` is given, else evaluates.

    Returns (mean_loss, accuracy) over the whole loader.
    """
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, total_correct, total_examples = 0.0, 0, 0
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train_mode:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    early_stopping_patience: int | None = None,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    """Train ``model`` with standard (non-DP) SGD-with-momentum-free Adam.

    ``eval_loader`` here is the target's holdout split (non-members) used
    purely to *measure* generalization and, optionally, to pick the best
    checkpoint via early stopping. This does not make the holdout set any
    less of a "non-member": the model's weights are never updated from its
    gradients, only selected based on its loss. That's the standard
    convention in the MIA literature (Yeom et al., Shokri et al. both do
    this) — the alternative, carving out a *third* held-out validation
    split just for early stopping, would shrink our already-limited holdout
    set for no real gain in rigor.

    Returns the (possibly early-stopped) model and a per-epoch history list.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_state = None
    best_eval_loss = float("inf")
    patience_left = early_stopping_patience
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, device, criterion, optimizer)
        eval_loss, eval_acc = _run_epoch(model, eval_loader, device, criterion, optimizer=None)
        gap = train_acc - eval_acc

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "gap": gap,
            }
        )
        logger.info(
            "epoch %d/%d | train_loss=%.4f train_acc=%.4f | eval_loss=%.4f eval_acc=%.4f | gap=%.4f",
            epoch, epochs, train_loss, train_acc, eval_loss, eval_acc, gap,
        )

        if early_stopping_patience is not None:
            if eval_loss < best_eval_loss - 1e-4:
                best_eval_loss = eval_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_left = early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info("Early stopping at epoch %d (best eval_loss=%.4f).", epoch, best_eval_loss)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def train_model_dp(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    target_epsilon: float,
    target_delta: float,
    max_grad_norm: float = 1.0,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, float]]:
    """Train ``model`` with DP-SGD (Opacus) at a target epsilon.

    Returns (model, history, privacy_spent). ``privacy_spent`` records the
    epsilon actually achieved, which can differ slightly from the target
    because the accountant solves for noise on a discrete grid.

    Two deliberate differences from ``train_model``:

    * **No early stopping.** Peeking at held-out loss each epoch to choose a
      checkpoint is a data-dependent decision that sits outside the privacy
      accounting — the reported epsilon would no longer describe the whole
      procedure. We train for a fixed number of epochs instead.
    * **No weight decay.** It interacts with per-sample gradient clipping in
      ways that muddy the comparison across epsilon values; the sweep should
      isolate the effect of noise, not of noise plus L2.

    ``target_epsilon=inf`` is accepted and falls through to ordinary
    non-private training, so the Phase 6 sweep can treat "no DP" as just
    another point on the epsilon axis instead of special-casing it.
    """
    from . import defenses

    if not np.isfinite(target_epsilon):
        logger.info("target_epsilon=inf -> training without DP (baseline point in the sweep).")
        model, history = train_model(
            model, train_loader, eval_loader, device, epochs=epochs, lr=lr,
        )
        return model, history, {"epsilon": float("inf"), "delta": target_delta}

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model, optimizer, train_loader, privacy_engine = defenses.make_private(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )

    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, device, criterion, optimizer)
        eval_loss, eval_acc = _run_epoch(model, eval_loader, device, criterion, optimizer=None)
        epsilon_so_far = privacy_engine.get_epsilon(target_delta)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "gap": train_acc - eval_acc,
                "epsilon_spent": epsilon_so_far,
            }
        )
        logger.info(
            "[DP] epoch %d/%d | train_acc=%.4f eval_acc=%.4f | gap=%.4f | eps_spent=%.3f",
            epoch, epochs, train_acc, eval_acc, train_acc - eval_acc, epsilon_so_far,
        )

    privacy_spent = {
        "epsilon": float(privacy_engine.get_epsilon(target_delta)),
        "delta": float(target_delta),
        "noise_multiplier": float(optimizer.noise_multiplier),
        "max_grad_norm": float(max_grad_norm),
    }

    # Unwrap Opacus's GradSampleModule so the saved checkpoint and downstream
    # attack code see a plain nn.Module with the original parameter names.
    unwrapped = model._module if hasattr(model, "_module") else model
    return unwrapped, history, privacy_spent


@torch.no_grad()
def compute_per_example_stats(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Compute per-example loss/confidence/probabilities for every item in
    ``dataset``, in dataset order.

    This is the raw material every attack in this lab consumes: the
    loss-threshold baseline (Phase 2) needs ``loss``, the shadow attack
    (Phase 3) needs the full ``probs`` vector, and LiRA (Phase 4) needs
    logit-scaled confidence. Computing and saving all of it once here means
    later phases just load an .npz instead of re-running inference.
    """
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="none")

    losses, confidences, predicted, true_labels, max_probs = [], [], [], [], []
    all_probs, all_logits = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, dim=1)
        conf_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        losses.append(loss.cpu().numpy())
        confidences.append(conf_true.cpu().numpy())
        predicted.append(probs.argmax(dim=1).cpu().numpy())
        true_labels.append(labels.cpu().numpy())
        max_probs.append(probs.max(dim=1).values.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        # Raw logits are kept as well as probs: LiRA's logit-scaling transform
        # is only numerically stable when computed from logits directly
        # (softmax probabilities saturate to exactly 1.0 in float32 on a
        # memorized training example, which would blow up to infinity).
        all_logits.append(logits.cpu().numpy())

    return {
        "loss": np.concatenate(losses),
        "confidence_true_label": np.concatenate(confidences),
        "predicted_label": np.concatenate(predicted),
        "true_label": np.concatenate(true_labels),
        "max_prob": np.concatenate(max_probs),
        "probs": np.concatenate(all_probs, axis=0),
        "logits": np.concatenate(all_logits, axis=0),
    }
