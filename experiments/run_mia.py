"""Run every membership inference attack against a trained target and emit a
comparison table, ROC plot, and score-distribution plot.

Covers Phase 2 (loss/confidence threshold), Phase 3 (shadow-model attack,
global and per-class), and Phase 4 (LiRA).

Usage:
    python experiments/run_mia.py --config configs/mia_overfit.yaml
    python experiments/run_mia.py --config configs/mia_overfit.yaml --skip-lira

Requires that the target was already trained by experiments/train_targets.py
(it reads that run's saved checkpoint and per-example artifacts).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, metrics, models, plots, train, utils
from src.attacks import lira, loss_threshold, shadow_mia

logger = logging.getLogger(__name__)


def _calibrate_thresholds_on_aux(
    pool, splits, config: dict, device: torch.device
) -> dict[str, float]:
    """Train one shadow model on the attacker's auxiliary data and derive a
    calibrated threshold for every score function from it.

    This is the honest way to calibrate: the attacker has labelled auxiliary
    data (disjoint from the target's evaluation splits by construction), so
    they train their own model on half of it, observe how the score separates
    *their* members from *their* non-members, and transfer the threshold. They
    never touch the target's member labels.

    One model serves every score function — the shadow's in/out statistics are
    computed once and each score is just a different view of them, so training
    a second model per score would be pure waste.

    Caveat worth knowing: a threshold transfers across models only as well as
    the two models' loss *scales* match. That's why the shadow is trained with
    the target's recipe, and it is also why AUC/TPR@FPR (threshold-free) are
    the metrics this lab leads with.
    """
    aux_indices = splits.attacker_aux
    aux_train_size = len(aux_indices) // 2
    in_indices, out_indices = data.make_in_out_split(aux_indices, aux_train_size, seed=config["seed"])

    utils.set_seed(config["seed"] + 77)
    shadow_model = models.build_model(**config["model"])
    in_subset, out_subset = data.get_subset(pool, in_indices), data.get_subset(pool, out_indices)

    logger.info("Calibrating thresholds: training one shadow model on %d aux examples", len(in_indices))
    shadow_model, _ = train.train_model(
        shadow_model,
        DataLoader(in_subset, batch_size=config["train"]["batch_size"], shuffle=True),
        DataLoader(out_subset, batch_size=config["train"]["batch_size"], shuffle=False),
        device,
        epochs=config["train"]["epochs"],
        lr=config["train"]["lr"],
        weight_decay=config["train"].get("weight_decay", 0.0),
        label_smoothing=config["train"].get("label_smoothing", 0.0),
        early_stopping_patience=config["train"].get("early_stopping_patience"),
    )

    shadow_in_stats = train.compute_per_example_stats(shadow_model, in_subset, device)
    shadow_out_stats = train.compute_per_example_stats(shadow_model, out_subset, device)

    return {
        score_name: loss_threshold.choose_threshold(
            score_fn(shadow_in_stats), score_fn(shadow_out_stats)
        )
        for score_name, score_fn in loss_threshold.SCORE_FUNCTIONS.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-lira", action="store_true",
                        help="Skip Phase 4 (LiRA), which is the most expensive attack.")
    parser.add_argument("--skip-shadow", action="store_true",
                        help="Skip Phase 3 shadow models (useful for a fast Phase 2-only run).")
    args = parser.parse_args()

    utils.setup_logging()
    config = utils.load_config(args.config)
    utils.set_seed(config["seed"])
    device = utils.get_device()

    target_run = config["target_run_name"]
    results_dir = Path(config["results_dir"])
    logger.info("Attacking target '%s' on device %s", target_run, device)

    # The target's per-example outputs were saved at training time.
    member_stats = utils.load_arrays(results_dir / "logits" / f"{target_run}_members.npz")
    nonmember_stats = utils.load_arrays(results_dir / "logits" / f"{target_run}_nonmembers.npz")

    pool = data.get_cifar10_pool(config["data"]["data_dir"])
    splits = data.make_splits(
        pool_size=len(pool),
        target_train_size=config["data"]["target_train_size"],
        target_holdout_size=config["data"]["target_holdout_size"],
        shadow_pool_size=config["data"]["shadow_pool_size"],
        attacker_aux_size=config["data"]["attacker_aux_size"],
        seed=config["seed"],
    )

    table_rows: list[dict] = []
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ---------- Phase 2: threshold attacks ----------
    thresholds = _calibrate_thresholds_on_aux(pool, splits, config, device)
    for score_name in ("loss", "confidence"):
        result = loss_threshold.run_threshold_attack(
            member_stats, nonmember_stats, score_name=score_name,
            threshold=thresholds[score_name], with_ci=True,
        )
        attack_label = f"threshold ({score_name})"
        table_rows.append({"attack": attack_label, **result.metrics})
        logger.info("%s -> %s", attack_label, result.metrics)

        # Only the loss variant is plotted. The confidence variant is
        # rank-equivalent to it (loss == -log confidence), so its ROC is
        # pixel-identical and would be drawn invisibly underneath, costing a
        # legend slot and a palette hue to show nothing. Both still appear in
        # the table, where their differing thresholds do matter.
        if score_name == "loss":
            roc_curves["threshold (loss / confidence)"] = metrics.roc_curve_points(
                result.scores, result.labels
            )
            n_members = len(member_stats["loss"])
            plots.plot_score_distributions(
                result.scores[:n_members], result.scores[n_members:],
                results_dir / "plots" / f"{target_run}_score_distributions.png",
                score_label="Membership score (negated loss)",
                title=f"{target_run}: member vs non-member loss",
            )

    # ---------- Phase 3: shadow-model attack ----------
    if not args.skip_shadow:
        shadow_runs = shadow_mia.train_shadow_models(
            pool=pool,
            candidate_indices=splits.shadow_pool,
            num_shadows=config["shadow"]["num_shadows"],
            shadow_train_size=config["shadow"]["shadow_train_size"],
            model_config=config["model"],
            train_config=config["train"],
            device=device,
            seed=config["seed"],
        )

        for per_class in (False, True):
            scores, labels, attack_metrics_dict = shadow_mia.run_shadow_attack(
                shadow_runs,
                member_stats,
                nonmember_stats,
                classifier_kind=config["shadow"]["classifier"],
                per_class=per_class,
                with_ci=True,
            )
            attack_label = f"shadow MIA ({'per-class' if per_class else 'global'})"
            table_rows.append({"attack": attack_label, **attack_metrics_dict})
            roc_curves[attack_label] = metrics.roc_curve_points(scores, labels)
            logger.info("%s -> %s", attack_label, attack_metrics_dict)

    # ---------- Phase 4: LiRA ----------
    if not args.skip_lira:
        # The challenge set is members-then-non-members, matching the order
        # run_lira_attack concatenates the target statistics in.
        challenge_indices = np.concatenate([splits.target_train, splits.target_holdout])
        shadow_outputs = lira.train_lira_shadows(
            pool=pool,
            challenge_indices=challenge_indices,
            num_shadows=config["lira"]["num_shadows"],
            model_config=config["model"],
            train_config=config["train"],
            device=device,
            in_fraction=config["lira"].get("in_fraction", 0.5),
            seed=config["seed"],
        )
        scores, labels, attack_metrics_dict = lira.run_lira_attack(
            shadow_outputs, member_stats, nonmember_stats,
            variance_mode=config["lira"].get("variance_mode", "global"),
            with_ci=True,
        )
        table_rows.append({"attack": "LiRA (simplified)", **attack_metrics_dict})
        roc_curves["LiRA (simplified)"] = metrics.roc_curve_points(scores, labels)
        logger.info("LiRA -> %s", attack_metrics_dict)

    # ---------- Report ----------
    table = pd.DataFrame(table_rows)
    table_path = results_dir / "tables" / f"{target_run}_mia_comparison.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)

    plots.plot_roc_loglog(
        roc_curves,
        results_dir / "plots" / f"{target_run}_mia_roc.png",
        title=f"{target_run}: membership inference ROC",
    )
    utils.save_json(
        {"target_run": target_run, "attacks": table_rows},
        results_dir / "metrics" / f"{target_run}_mia.json",
    )

    print(f"\n=== MIA comparison for '{target_run}' ===")
    display = table.copy()

    def _fold_interval(column: str, precision: int) -> None:
        """Render "estimate [low, high]" in one cell, if the CIs were computed.

        Column access is by label rather than via ``itertuples``: pandas
        renames attributes that aren't valid Python identifiers, and these
        column names contain dots ("tpr_at_fpr_0.001"), so attribute access
        silently gets the wrong field.
        """
        low, high = f"{column}_ci_low", f"{column}_ci_high"
        if low not in display.columns:
            return
        display[column] = [
            f"{value:.{precision}f} [{lo:.{precision}f}, {hi:.{precision}f}]"
            for value, lo, hi in zip(table[column], table[low], table[high])
        ]

    _fold_interval("auc", 4)
    _fold_interval("tpr_at_fpr_0.001", 4)

    display_columns = [c for c in ("attack", "auc", "balanced_accuracy",
                                   "tpr_at_fpr_0.01", "tpr_at_fpr_0.001") if c in display.columns]
    print(display[display_columns].to_string(index=False))
    print(
        "\nNote: the two threshold rows are rank-equivalent by construction "
        "(loss == -log confidence), so their AUC and TPR@FPR are necessarily "
        "identical; only their calibrated thresholds differ. Read them as one "
        "attack, not two."
    )
    print(f"\nSaved table -> {table_path}")


if __name__ == "__main__":
    main()
