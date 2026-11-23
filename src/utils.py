"""Cross-cutting helpers used by every script in the lab: reproducible
seeding, YAML config loading, device selection, and artifact (JSON / npz)
persistence.

Centralizing these means every experiment script does the same thing the
same way, which matters a lot for a lab whose entire point is comparing
numbers across runs.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch RNGs and force deterministic cuDNN kernels.

    Determinism costs a little speed (cuDNN can't autotune to the fastest,
    possibly non-deterministic, algorithm) but this lab's whole purpose is
    comparing runs against each other, so reproducibility wins.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA if available, else CPU.

    Every script calls this instead of hard-coding "cuda" so the lab runs
    unchanged on a laptop CPU or a free Colab GPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config, resolving one optional ``extends:``
    key that names a base config file (resolved relative to ``path``'s
    directory). Keys in the child override keys in the base at the
    top level.

    We only support this single level of inheritance rather than adopting
    Hydra: Hydra's composition/override syntax is powerful but is a real
    learning curve, and every config here is small enough that "child
    overrides base, one level deep" covers all of this lab's needs.
    """
    path = Path(path)
    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}

    base_name = config.pop("extends", None)
    if base_name is None:
        return config

    base_config = load_config(path.parent / base_name)
    merged = dict(base_config)
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the log format used by every script in the lab."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dict as pretty-printed JSON, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_arrays(path: str | Path, **arrays: np.ndarray) -> None:
    """Save named numpy arrays (e.g. per-example loss/logits/labels) to one
    .npz file, so a downstream attack can load everything it needs about a
    model's predictions on a split with a single call.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
