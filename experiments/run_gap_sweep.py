"""Phase 2 headline experiment: does overfitting cause membership leakage?

Trains a grid of targets with varying epoch counts and training-set sizes,
attacks each with the (cheap) loss-threshold attack, and plots the
generalization gap against attack success.

Usage:
    python experiments/run_gap_sweep.py --config configs/gap_sweep.yaml

Each (epochs, train_size, seed) combination trains one model, so the grid
size multiplies quickly — keep it small on CPU and widen it on a GPU.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, metrics, models, plots, train, utils
from src.attacks import loss_threshold

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    utils.setup_logging()
    config = utils.load_config(args.config)
    device = utils.get_device()
    results_dir = Path(config["results_dir"])

    pool = data.get_cifar10_pool(config["data"]["data_dir"])
    sweep = config["sweep"]
    combinations = list(itertools.product(sweep["epochs"], sweep["train_sizes"], sweep["seeds"]))
    logger.info("Sweep: %d models to train (%d epochs x %d sizes x %d seeds)",
                len(combinations), len(sweep["epochs"]), len(sweep["train_sizes"]), len(sweep["seeds"]))

    rows = []
    for epochs, train_size, seed in combinations:
        utils.set_seed(seed)
        # Re-split per seed so that a "seed" varies both the data partition and
        # the weight init — otherwise the error bars only capture init noise and
        # understate real run-to-run variation.
        splits = data.make_splits(
            pool_size=len(pool),
            target_train_size=train_size,
            # Holdout matched to train size keeps the MIA evaluation balanced.
            target_holdout_size=train_size,
            shadow_pool_size=config["data"]["shadow_pool_size"],
            attacker_aux_size=config["data"]["attacker_aux_size"],
            seed=seed,
        )
        member_subset = data.get_subset(pool, splits.target_train)
        nonmember_subset = data.get_subset(pool, splits.target_holdout)

        model = models.build_model(**config["model"])
        logger.info("Training: epochs=%d train_size=%d seed=%d", epochs, train_size, seed)
        model, history = train.train_model(
            model,
            DataLoader(member_subset, batch_size=config["train"]["batch_size"], shuffle=True),
            DataLoader(nonmember_subset, batch_size=config["train"]["batch_size"], shuffle=False),
            device,
            epochs=epochs,
            lr=config["train"]["lr"],
            weight_decay=config["train"].get("weight_decay", 0.0),
        )

        member_stats = train.compute_per_example_stats(model, member_subset, device)
        nonmember_stats = train.compute_per_example_stats(model, nonmember_subset, device)
        result = loss_threshold.run_threshold_attack(member_stats, nonmember_stats, score_name="loss")

        rows.append(
            {
                "epochs": epochs,
                "train_size": train_size,
                "seed": seed,
                "train_acc": history[-1]["train_acc"],
                "test_acc": history[-1]["eval_acc"],
                "generalization_gap": history[-1]["gap"],
                "auc": result.metrics["auc"],
                "tpr_at_fpr_0.01": result.metrics["tpr_at_fpr_0.01"],
                "tpr_at_fpr_0.001": result.metrics["tpr_at_fpr_0.001"],
            }
        )

    frame = pd.DataFrame(rows)
    table_path = results_dir / "tables" / "gap_sweep.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table_path, index=False)

    # Aggregate over seeds so the plotted points carry mean +/- std, per the
    # lab's evaluation rules, instead of showing one noisy run per setting.
    grouped = (
        frame.groupby(["epochs", "train_size"])
        .agg(gap_mean=("generalization_gap", "mean"), gap_std=("generalization_gap", "std"),
             auc_mean=("auc", "mean"), auc_std=("auc", "std"),
             tpr_mean=("tpr_at_fpr_0.001", "mean"), tpr_std=("tpr_at_fpr_0.001", "std"))
        .reset_index()
    )
    # A single seed makes pandas' sample std NaN; matplotlib can't draw that,
    # so fall back to zero-length bars rather than failing the whole run.
    grouped[["gap_std", "auc_std", "tpr_std"]] = grouped[["gap_std", "auc_std", "tpr_std"]].fillna(0.0)
    grouped.to_csv(results_dir / "tables" / "gap_sweep_aggregated.csv", index=False)

    point_labels = [f"e{int(r.epochs)}/n{int(r.train_size)}" for r in grouped.itertuples()]
    plots.plot_gap_vs_attack(
        grouped["gap_mean"].tolist(),
        grouped["auc_mean"].tolist(),
        results_dir / "plots" / "gap_vs_attack_auc.png",
        point_labels=point_labels,
        attack_metric_label="Attack AUC (loss threshold)",
        gap_errors=grouped["gap_std"].tolist(),
        attack_errors=grouped["auc_std"].tolist(),
    )
    plots.plot_gap_vs_attack(
        grouped["gap_mean"].tolist(),
        grouped["tpr_mean"].tolist(),
        results_dir / "plots" / "gap_vs_attack_tpr.png",
        point_labels=point_labels,
        attack_metric_label="TPR @ 0.1% FPR (loss threshold)",
        title="Generalization gap vs worst-case leakage",
        gap_errors=grouped["gap_std"].tolist(),
        attack_errors=grouped["tpr_std"].tolist(),
    )

    print("\n=== Gap sweep (aggregated over seeds) ===")
    print(grouped.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved -> {table_path}")


if __name__ == "__main__":
    main()
