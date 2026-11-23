"""CIFAR-10 loading and the disjoint splits the rest of the lab is built on.

Every later phase depends on these four splits never overlapping:

- ``target_train``    the target model's members (it trains on these)
- ``target_holdout``  non-members: same distribution, never trained on
- ``shadow_pool``     data shadow models (Phase 3) are trained from
- ``attacker_aux``    data the attacker calibrates baseline attacks on

If ``attacker_aux`` overlapped with ``target_holdout``, a threshold "tuned
on auxiliary data" would secretly be tuned on the real evaluation targets,
which would make every attack look stronger than it really is. We avoid
that entirely by construction: all four splits are non-overlapping slices
of one seeded random permutation of the full data pool, so disjointness
holds by design rather than by convention (and is still unit-tested).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import datasets, transforms

# Standard CIFAR-10 per-channel mean/std (computed over the training set).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transform(augment: bool = False) -> transforms.Compose:
    """Build the CIFAR-10 preprocessing pipeline.

    ``augment=False`` (the default) is what member/non-member evaluation
    always uses — augmenting eval data would make loss/confidence values
    incomparable between the two groups. Training may optionally set
    ``augment=True`` for its own transform.
    """
    ops = []
    if augment:
        ops += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    return transforms.Compose(ops)


def get_cifar10_pool(data_dir: str | Path, augment: bool = False) -> Dataset:
    """Load CIFAR-10 train+test concatenated into one 60,000-image pool.

    We combine both official splits because *we* define what's a member vs.
    non-member for this lab — torchvision's train/test boundary is not
    meaningful here. Concatenation order (train then test) is fixed, so an
    index into this pool always refers to the same image across calls,
    which is what makes seeded-permutation splitting in ``make_splits``
    reproducible.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    transform = build_transform(augment=augment)
    train = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=transform)
    return ConcatDataset([train, test])


def get_cifar100_query_pool(data_dir: str | Path) -> Dataset:
    """Load CIFAR-100 as the attacker's unlabeled query pool (Phase 5).

    Normalized with **CIFAR-10's** statistics, not CIFAR-100's, because these
    images are fed to a model that was trained expecting CIFAR-10
    preprocessing — the attacker is querying someone else's deployed API and
    must match its input contract.

    Using CIFAR-100 rather than held-out CIFAR-10 is the honest choice: a real
    attacker has *some* vaguely related unlabeled data, not samples from the
    victim's exact training distribution. Giving the attacker in-distribution
    data would inflate the extraction results.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return datasets.CIFAR100(
        root=str(data_dir), train=True, download=True, transform=build_transform(augment=False)
    )


def get_subset(dataset: Dataset, indices: np.ndarray) -> Subset:
    """Wrap ``dataset`` down to just the given indices."""
    return Subset(dataset, indices.tolist())


@dataclass
class DatasetSplits:
    """Four disjoint index arrays into a CIFAR-10 pool. See module docstring."""

    target_train: np.ndarray
    target_holdout: np.ndarray
    shadow_pool: np.ndarray
    attacker_aux: np.ndarray


def make_splits(
    pool_size: int,
    target_train_size: int,
    target_holdout_size: int,
    shadow_pool_size: int,
    attacker_aux_size: int,
    seed: int,
) -> DatasetSplits:
    """Partition ``range(pool_size)`` into four disjoint splits.

    Implementation: shuffle a full index permutation once, then slice it
    sequentially. Slices of a permutation can never overlap, so
    disjointness is structural rather than something we have to carefully
    get right with set arithmetic.
    """
    total_requested = target_train_size + target_holdout_size + shadow_pool_size + attacker_aux_size
    if total_requested > pool_size:
        raise ValueError(
            f"Requested splits sum to {total_requested} examples, but the pool "
            f"only has {pool_size}. Reduce split sizes in your config."
        )

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(pool_size)

    a = target_train_size
    b = a + target_holdout_size
    c = b + shadow_pool_size
    d = c + attacker_aux_size

    return DatasetSplits(
        target_train=permutation[:a],
        target_holdout=permutation[a:b],
        shadow_pool=permutation[b:c],
        attacker_aux=permutation[c:d],
    )


def make_in_out_split(
    candidate_indices: np.ndarray,
    in_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``candidate_indices`` into an "IN" (trained-on) and "OUT" part.

    Used to give each shadow model its own known member/non-member division
    (Phase 3) and to build LiRA's random in/out draws (Phase 4). Each shadow
    gets a *different* ``seed`` so that, across many shadows, every candidate
    example lands IN roughly half the time — which is exactly the condition
    LiRA's per-example IN/OUT distributions need.
    """
    if in_size > len(candidate_indices):
        raise ValueError(
            f"in_size={in_size} exceeds the {len(candidate_indices)} available candidate indices."
        )
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(candidate_indices)
    return shuffled[:in_size], shuffled[in_size:]
