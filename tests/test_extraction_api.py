"""Tests for the black-box API: query accounting and output hardening.

Query counting is load-bearing for the whole Phase 5 result — the extraction
plot's x-axis is only meaningful if the counter is exact. Output hardening is
tested here too because a bug that silently returned un-hardened probabilities
would make every hardening defense look useless for the wrong reason.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from src import defenses
from src.attacks import extraction


class _ConstantLogitModel(nn.Module):
    """Returns fixed logits regardless of input, so outputs are predictable."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = nn.Parameter(logits, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits.unsqueeze(0).expand(x.shape[0], -1)


@pytest.fixture
def api() -> extraction.BlackBoxAPI:
    model = _ConstantLogitModel(torch.tensor([3.0, 2.0, 1.0, 0.0]))
    return extraction.BlackBoxAPI(model, torch.device("cpu"))


def test_query_count_starts_at_zero(api):
    assert api.query_count == 0


def test_query_count_counts_examples_not_calls(api):
    """One call with 8 examples must cost 8 queries, not 1.

    Otherwise an attacker defeats any budget by batching everything into a
    single call, and the lab's cost model would be fiction.
    """
    api.predict(torch.zeros(8, 3, 4, 4))
    assert api.query_count == 8

    api.predict(torch.zeros(5, 3, 4, 4))
    assert api.query_count == 13


def test_predict_labels_counts_queries_too(api):
    api.predict_labels(torch.zeros(4, 3, 4, 4))
    assert api.query_count == 4


def test_budget_enforced_and_rejects_overrun():
    model = _ConstantLogitModel(torch.tensor([1.0, 0.0]))
    api = extraction.BlackBoxAPI(model, torch.device("cpu"), query_budget=10)

    api.predict(torch.zeros(10, 3, 4, 4))
    assert api.queries_remaining == 0

    with pytest.raises(extraction.QueryBudgetExceeded):
        api.predict(torch.zeros(1, 3, 4, 4))


def test_budget_rejects_batch_that_would_overshoot():
    """A request larger than the remaining budget is refused outright, and
    must not partially consume the budget."""
    model = _ConstantLogitModel(torch.tensor([1.0, 0.0]))
    api = extraction.BlackBoxAPI(model, torch.device("cpu"), query_budget=5)

    with pytest.raises(extraction.QueryBudgetExceeded):
        api.predict(torch.zeros(6, 3, 4, 4))
    assert api.query_count == 0


def test_unmetered_api_reports_no_remaining_budget(api):
    assert api.queries_remaining is None


def test_reset_counter(api):
    api.predict(torch.zeros(3, 3, 4, 4))
    api.reset_counter()
    assert api.query_count == 0


def test_full_mode_returns_real_probabilities(api):
    response = api.predict(torch.zeros(2, 3, 4, 4))
    assert response.shape == (2, 4)
    assert torch.allclose(response.sum(dim=1), torch.ones(2), atol=1e-5)
    # Strictly decreasing, matching the constant logits [3, 2, 1, 0].
    assert response[0, 0] > response[0, 1] > response[0, 2] > response[0, 3]


def test_label_mode_returns_one_hot():
    model = _ConstantLogitModel(torch.tensor([3.0, 2.0, 1.0, 0.0]))
    api = extraction.BlackBoxAPI(model, torch.device("cpu"), output_mode="label")
    response = api.predict(torch.zeros(2, 3, 4, 4))
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(response, expected)


def test_top_k_truncation_zeroes_tail_and_renormalizes():
    model = _ConstantLogitModel(torch.tensor([3.0, 2.0, 1.0, 0.0]))
    api = extraction.BlackBoxAPI(model, torch.device("cpu"), top_k=2)
    response = api.predict(torch.zeros(1, 3, 4, 4))

    assert (response[0, 2:] == 0).all()
    assert response.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_harden_probabilities_rounding_renormalizes():
    probs = torch.tensor([[0.514, 0.286, 0.2]])
    hardened = defenses.harden_probabilities(probs, round_digits=1)
    assert hardened.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_harden_probabilities_degenerate_rounding_falls_back():
    """Rounding to 0 decimals can zero an entire row; the defense must fall
    back to the un-hardened row rather than emit NaNs."""
    probs = torch.tensor([[0.34, 0.33, 0.33]])
    hardened = defenses.harden_probabilities(probs, round_digits=0)
    assert torch.isfinite(hardened).all()
    assert hardened.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_harden_probabilities_rejects_bad_arguments():
    probs = torch.tensor([[0.5, 0.3, 0.2]])
    with pytest.raises(ValueError):
        defenses.harden_probabilities(probs, mode="nonsense")
    with pytest.raises(ValueError):
        defenses.harden_probabilities(probs, top_k=99)
    with pytest.raises(ValueError):
        defenses.harden_probabilities(probs, round_digits=-1)


def test_collect_api_responses_counts_exactly_num_queries():
    model = _ConstantLogitModel(torch.tensor([1.0, 0.0]))
    api = extraction.BlackBoxAPI(model, torch.device("cpu"))
    dataset = TensorDataset(torch.zeros(50, 3, 4, 4), torch.zeros(50, dtype=torch.long))

    stolen = extraction.collect_api_responses(api, dataset, num_queries=30, batch_size=8)
    assert api.query_count == 30
    assert len(stolen) == 30


def test_collect_api_responses_rejects_oversized_request():
    model = _ConstantLogitModel(torch.tensor([1.0, 0.0]))
    api = extraction.BlackBoxAPI(model, torch.device("cpu"))
    dataset = TensorDataset(torch.zeros(10, 3, 4, 4), torch.zeros(10, dtype=torch.long))

    with pytest.raises(ValueError):
        extraction.collect_api_responses(api, dataset, num_queries=99)


def test_soft_cross_entropy_matches_hard_ce_on_one_hot():
    """With a one-hot teacher, soft cross-entropy must reduce to standard CE."""
    student_logits = torch.tensor([[2.0, 1.0, 0.1]])
    teacher_probs = torch.tensor([[1.0, 0.0, 0.0]])
    soft = extraction.soft_cross_entropy(student_logits, teacher_probs)
    hard = torch.nn.functional.cross_entropy(student_logits, torch.tensor([0]))
    assert soft.item() == pytest.approx(hard.item(), abs=1e-6)


def test_apply_hardening_to_stats_label_mode_collapses_to_correctness():
    """Under a label-only API the attacker's loss signal must degrade to a
    single bit: correct (low loss) vs incorrect (high loss)."""
    stats = {
        "probs": np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]], dtype=np.float32),
        "true_label": np.array([0, 0]),
        "loss": np.array([0.36, 1.61], dtype=np.float32),
        "confidence_true_label": np.array([0.7, 0.2], dtype=np.float32),
        "max_prob": np.array([0.7, 0.7], dtype=np.float32),
        "predicted_label": np.array([0, 1]),
    }
    hardened = defenses.apply_hardening_to_stats(stats, mode="label")

    # Example 0 was predicted correctly -> confidence 1.0, loss ~0.
    assert hardened["confidence_true_label"][0] == pytest.approx(1.0)
    assert hardened["loss"][0] == pytest.approx(0.0, abs=1e-6)
    # Example 1 was predicted wrong -> true-class confidence 0, loss huge.
    assert hardened["confidence_true_label"][1] == pytest.approx(0.0)
    assert hardened["loss"][1] > 10
