"""Train one target model from a YAML config and save its checkpoint plus
per-example loss/logit artifacts for members (target_train) and non-members
(target_holdout).

Usage:
    python experiments/train_targets.py --config configs/target_overfit.yaml
    python experiments/train_targets.py --config configs/target_regularized.yaml

This script is intentionally thin: every real decision (splitting, model
construction, the training loop) lives in src/, so it can also be called
from a notebook or a future automated sweep without duplicating logic.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `python experiments/x.py` finds src/

from src import data, models, train, utils

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a target-model YAML config.")
    args = parser.parse_args()

    utils.setup_logging()
    config = utils.load_config(args.config)
    utils.set_seed(config["seed"])
    device = utils.get_device()
    logger.info("run_name=%s device=%s", config["run_name"], device)

    pool = data.get_cifar10_pool(config["data"]["data_dir"])
    splits = data.make_splits(
        pool_size=len(pool),
        target_train_size=config["data"]["target_train_size"],
        target_holdout_size=config["data"]["target_holdout_size"],
        shadow_pool_size=config["data"]["shadow_pool_size"],
        attacker_aux_size=config["data"]["attacker_aux_size"],
        seed=config["seed"],
    )

    num_workers = config["data"].get("num_workers", 0)
    train_loader = DataLoader(
        data.get_subset(pool, splits.target_train),
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
    )
    holdout_loader = DataLoader(
        data.get_subset(pool, splits.target_holdout),
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )

    model = models.build_model(**config["model"])

    model, history = train.train_model(
        model,
        train_loader,
        holdout_loader,
        device,
        epochs=config["train"]["epochs"],
        lr=config["train"]["lr"],
        weight_decay=config["train"].get("weight_decay", 0.0),
        label_smoothing=config["train"].get("label_smoothing", 0.0),
        early_stopping_patience=config["train"].get("early_stopping_patience"),
    )

    run_name = config["run_name"]
    results_dir = Path(config["results_dir"])

    checkpoint_path = results_dir / "checkpoints" / f"{run_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    logger.info("Saved checkpoint to %s", checkpoint_path)

    # Per-example artifacts for both members and non-members: these are what
    # every attack in Phase 2+ consumes, so we compute them once, here.
    member_stats = train.compute_per_example_stats(model, data.get_subset(pool, splits.target_train), device)
    nonmember_stats = train.compute_per_example_stats(model, data.get_subset(pool, splits.target_holdout), device)
    utils.save_arrays(results_dir / "logits" / f"{run_name}_members.npz", **member_stats)
    utils.save_arrays(results_dir / "logits" / f"{run_name}_nonmembers.npz", **nonmember_stats)

    final = history[-1]
    utils.save_json(
        {
            "run_name": run_name,
            "config_path": args.config,
            "history": history,
            "final_train_acc": final["train_acc"],
            "final_eval_acc": final["eval_acc"],
            "generalization_gap": final["gap"],
        },
        results_dir / "metrics" / f"{run_name}.json",
    )

    logger.info(
        "Done. final_train_acc=%.4f final_eval_acc=%.4f generalization_gap=%.4f",
        final["train_acc"], final["eval_acc"], final["gap"],
    )


if __name__ == "__main__":
    main()
