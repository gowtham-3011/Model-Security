"""Phase 5: black-box model extraction (model stealing).

Threat model: the attacker cannot see the target's weights, architecture, or
training data. They can only call ``predict()`` on a deployed API, and each
call costs them something (money, rate limit, detection risk) — so the
interesting question is not "can you steal it?" but **"how much does it cost
per unit of fidelity?"**

The attack itself is knowledge distillation with a stolen teacher: query the
target on whatever unlabeled data the attacker has, treat the responses as
labels, and train a surrogate on those pairs. Note the attacker never needs
the *true* labels of their query data — the target supplies them. That is why
extraction is cheap and why an "unlabeled data" requirement is such a low bar.

Two things make this lab's version realistic:

* The attacker's query data is **CIFAR-100**, not CIFAR-10. A real attacker
  rarely has in-distribution data; they have *something* vaguely similar.
  Using the target's own training distribution would quietly overstate the
  attack.
* The API **counts and can cap queries**, so the query-budget curve the lab
  plots reflects a real constraint rather than an afterthought.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .. import defenses, metrics

logger = logging.getLogger(__name__)


class QueryBudgetExceeded(RuntimeError):
    """Raised when an attacker exceeds the API's configured query budget."""


class BlackBoxAPI:
    """A deployed model exposed as a query-counting black box.

    This class is the lab's enforcement of the black-box threat model: it
    holds the model privately and exposes exactly one capability,
    ``predict()``. Attack code is given *this object*, never the underlying
    model, so an attack cannot accidentally cheat by reading weights,
    gradients, or the training set — a mistake that is easy to make and
    silently invalidates extraction results.

    Output hardening (``output_mode``, ``top_k``, ``round_digits``) is applied
    here rather than in the attack, because that is where it lives in
    reality: it is the *defender's* configuration choice. The same knobs are
    measured against membership inference in Phase 6.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_mode: str = "full",
        top_k: int | None = None,
        round_digits: int | None = None,
        query_budget: int | None = None,
        batch_size: int = 256,
    ):
        self._model = model.to(device).eval()
        self._device = device
        self.output_mode = output_mode
        self.top_k = top_k
        self.round_digits = round_digits
        self.query_budget = query_budget
        self.batch_size = batch_size
        self._query_count = 0

    @property
    def query_count(self) -> int:
        """Total number of individual examples served so far."""
        return self._query_count

    @property
    def queries_remaining(self) -> int | None:
        """Remaining budget, or None if the API is unmetered."""
        if self.query_budget is None:
            return None
        return max(0, self.query_budget - self._query_count)

    def reset_counter(self) -> None:
        """Reset the query counter (used between sweep points, not by attackers)."""
        self._query_count = 0

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Serve a batch of queries, returning (possibly hardened) probabilities.

        Counts one query *per example*, not per call — otherwise an attacker
        could trivially defeat the budget by batching everything into a single
        call, and the cost model would be meaningless.
        """
        num_queries = inputs.shape[0]
        if self.query_budget is not None and self._query_count + num_queries > self.query_budget:
            raise QueryBudgetExceeded(
                f"Request for {num_queries} queries would exceed the API budget: "
                f"{self._query_count}/{self.query_budget} already used."
            )

        probs = F.softmax(self._model(inputs.to(self._device)), dim=1)
        self._query_count += num_queries
        return defenses.harden_probabilities(
            probs, mode=self.output_mode, top_k=self.top_k, round_digits=self.round_digits
        )

    @torch.no_grad()
    def predict_labels(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning argmax labels. Counts queries identically."""
        return self.predict(inputs).argmax(dim=1)


def collect_api_responses(
    api: BlackBoxAPI,
    query_dataset: Dataset,
    num_queries: int,
    batch_size: int = 256,
) -> TensorDataset:
    """Query ``api`` on the first ``num_queries`` examples of the attacker's data.

    Returns a TensorDataset of (input, response) pairs — the attacker's
    stolen training set. Inputs are kept on CPU so the surrogate's DataLoader
    behaves normally.
    """
    if num_queries > len(query_dataset):
        raise ValueError(
            f"Requested {num_queries} queries but the attacker's query pool only has "
            f"{len(query_dataset)} examples. Enlarge the pool or lower the budget."
        )

    loader = DataLoader(torch.utils.data.Subset(query_dataset, range(num_queries)),
                       batch_size=batch_size, shuffle=False)

    input_blocks, response_blocks = [], []
    for images, _ in loader:  # the attacker ignores any true labels they happen to have
        input_blocks.append(images)
        response_blocks.append(api.predict(images).cpu())

    return TensorDataset(torch.cat(input_blocks), torch.cat(response_blocks))


def soft_cross_entropy(student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a soft target distribution.

    ``-sum_c p_teacher(c) * log p_student(c)``, averaged over the batch.

    This is what makes full-probability APIs so much more valuable to an
    attacker than label-only ones: a soft target carries information about
    *every* class, including the target's relative uncertainty between
    runner-ups. That signal describes the decision boundary's shape, not just
    which side of it a point falls on — so each query teaches the surrogate
    far more. ``F.cross_entropy`` can't be used directly because it expects
    hard integer labels.
    """
    return -(teacher_probs * F.log_softmax(student_logits, dim=1)).sum(dim=1).mean()


def train_surrogate(
    surrogate: nn.Module,
    stolen_dataset: TensorDataset,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int = 128,
    label_only: bool = False,
) -> nn.Module:
    """Train the attacker's surrogate on the target's responses.

    ``label_only=True`` collapses the responses to hard argmax labels, which
    is what the attacker is forced to do when the API returns only a top-1
    label. Keeping both paths in one function makes the two API settings
    directly comparable — the only thing that changes is the loss.
    """
    surrogate = surrogate.to(device).train()
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=lr)
    loader = DataLoader(stolen_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        total_loss, total_examples = 0.0, 0
        for images, responses in loader:
            images, responses = images.to(device), responses.to(device)
            optimizer.zero_grad()
            logits = surrogate(images)

            if label_only:
                loss = F.cross_entropy(logits, responses.argmax(dim=1))
            else:
                loss = soft_cross_entropy(logits, responses)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.shape[0]
            total_examples += images.shape[0]

        logger.info("surrogate epoch %d/%d | loss=%.4f", epoch, epochs, total_loss / total_examples)

    return surrogate


@dataclass
class ExtractionResult:
    """Outcome of one extraction run at a given query budget."""

    num_queries: int
    output_mode: str
    fidelity: float
    surrogate_accuracy: float
    target_accuracy: float


@torch.no_grad()
def evaluate_extraction(
    surrogate: nn.Module,
    api: BlackBoxAPI,
    eval_dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[float, float, float]:
    """Measure (fidelity, surrogate_accuracy, target_accuracy) on held-out data.

    Fidelity — agreement with the target — is the metric that says whether
    extraction *worked*. Surrogate accuracy says whether the attacker got a
    useful classifier, which is a different question: a surrogate can be
    accurate while disagreeing with the target often (it learned the task,
    not the model). Both are reported because the gap between them is
    informative.

    Queries made here are for *evaluation* and would not be available to a
    real attacker, so they are excluded from the attack's query budget by
    restoring the counter afterwards.
    """
    surrogate = surrogate.to(device).eval()
    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    queries_before = api.query_count

    surrogate_predictions, target_predictions, true_labels = [], [], []
    for images, labels in loader:
        surrogate_predictions.append(surrogate(images.to(device)).argmax(dim=1).cpu().numpy())
        target_predictions.append(api.predict(images).argmax(dim=1).cpu().numpy())
        true_labels.append(labels.numpy())

    api._query_count = queries_before  # evaluation is not part of the attack's budget

    surrogate_predictions = np.concatenate(surrogate_predictions)
    target_predictions = np.concatenate(target_predictions)
    true_labels = np.concatenate(true_labels)

    return (
        metrics.fidelity(surrogate_predictions, target_predictions),
        metrics.accuracy(surrogate_predictions, true_labels),
        metrics.accuracy(target_predictions, true_labels),
    )
