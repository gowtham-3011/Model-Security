"""Unit tests for src.data.make_splits.

The single most important invariant in this whole lab is that the target's
member/non-member splits, the shadow pool, and the attacker's auxiliary
data never overlap. Any leakage between them would let an "attack" cheat by
memorizing evaluation data instead of genuinely inferring membership, which
would silently invalidate every result in later phases.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data import make_splits

SPLIT_KWARGS = dict(
    pool_size=1000,
    target_train_size=200,
    target_holdout_size=200,
    shadow_pool_size=400,
    attacker_aux_size=150,
    seed=0,
)


def test_splits_are_pairwise_disjoint():
    splits = make_splits(**SPLIT_KWARGS)
    groups = [splits.target_train, splits.target_holdout, splits.shadow_pool, splits.attacker_aux]

    seen: set[int] = set()
    for group in groups:
        group_set = set(group.tolist())
        overlap = group_set & seen
        assert not overlap, f"found overlapping indices across splits: {overlap}"
        seen |= group_set


def test_split_sizes_match_request():
    splits = make_splits(**SPLIT_KWARGS)
    assert len(splits.target_train) == SPLIT_KWARGS["target_train_size"]
    assert len(splits.target_holdout) == SPLIT_KWARGS["target_holdout_size"]
    assert len(splits.shadow_pool) == SPLIT_KWARGS["shadow_pool_size"]
    assert len(splits.attacker_aux) == SPLIT_KWARGS["attacker_aux_size"]


def test_indices_within_pool_bounds():
    splits = make_splits(**SPLIT_KWARGS)
    for group in [splits.target_train, splits.target_holdout, splits.shadow_pool, splits.attacker_aux]:
        assert group.min() >= 0
        assert group.max() < SPLIT_KWARGS["pool_size"]


def test_same_seed_gives_same_splits():
    a = make_splits(**SPLIT_KWARGS)
    b = make_splits(**SPLIT_KWARGS)
    np.testing.assert_array_equal(a.target_train, b.target_train)
    np.testing.assert_array_equal(a.target_holdout, b.target_holdout)
    np.testing.assert_array_equal(a.shadow_pool, b.shadow_pool)
    np.testing.assert_array_equal(a.attacker_aux, b.attacker_aux)


def test_different_seed_gives_different_splits():
    a = make_splits(**SPLIT_KWARGS)
    b = make_splits(**{**SPLIT_KWARGS, "seed": 1})
    assert not np.array_equal(a.target_train, b.target_train)


def test_oversized_request_raises_value_error():
    with pytest.raises(ValueError):
        make_splits(
            pool_size=100,
            target_train_size=50,
            target_holdout_size=50,
            shadow_pool_size=50,
            attacker_aux_size=50,
            seed=0,
        )
