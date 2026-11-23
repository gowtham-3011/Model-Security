"""Phase 6: evaluate every defense against both attack families.

Three stages, all driven by one config:

1. **Cheap defenses** (retrain): weight decay, dropout, label smoothing,
   early stopping.
2. **Output hardening** (no retraining — it only changes the API): label-only,
   top-k truncation, rounded confidences. Measured against *both* MIA and
   extraction.
3. **DP-SGD** at each target epsilon, including epsilon=infinity as the
   non-private baseline.

Produces the lab's core privacy-utility plot and the master comparison table.

Usage:
    python experiments/run_defenses.py --config configs/defenses.yaml
    python experiments/run_defenses.py --config configs/defenses.yaml --skip-extraction
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

from src import data, defenses, models, plots, train, utils
from src.attacks import extraction, loss_threshold

logger = logging.getLogger(__name__)


def _measure_extraction_fidelity(
    target_model: torch.nn.Module,
    eval_dataset,
    query_pool,
    config: dict,
    device: torch.device,
    hardening: dict,
) -> float:
    """Run one extraction attempt at a fixed query budget and return fidelity."""
    api = extraction.BlackBoxAPI(
        target_model,
        device,
        output_mode=hardening.get("mode", "full"),
        top_k=hardening.get("top_k"),
        round_digits=hardening.get("round_digits"),
    )
    num_queries = config["defenses"]["extraction_query_budget"]
    stolen = extraction.collect_api_responses(api, query_pool, num_queries)

    utils.set_seed(config["seed"])
    surrogate = models.build_model(**config["defenses"]["surrogate_model"])
    surrogate = extraction.train_surrogate(
        surrogate, stolen, device,
        epochs=config["defenses"]["surrogate_epochs"],
        lr=config["defenses"]["surrogate_lr"],
        batch_size=config["train"]["batch_size"],
        label_only=hardening.get("mode") == "label",
    )
    fidelity, _, _ = extraction.evaluate_extraction(surrogate, api, eval_dataset, device)
    return fidelity


def _evaluate_defense(
    name: str,
    member_stats: dict,
    nonmember_stats: dict,
    test_accuracy: float,
    generalization_gap: float,
    extraction_fidelity: float | None,
    hardening: dict | None = None,
) -> dict:
    """Score one defended configuration on the lab's full metric bundle.

    MIA here is the loss-threshold attack: it needs no extra model training,
    so it can be run against every defense cheaply. AUC and TPR@low-FPR are
    threshold-free, so no calibration step is required to compare defenses
    fairly against each other.
    """
    if hardening:
        member_stats = defenses.apply_hardening_to_stats(member_stats, **hardening)
        nonmember_stats = defenses.apply_hardening_to_stats(nonmember_stats, **hardening)

    result = loss_threshold.run_threshold_attack(member_stats, nonmember_stats, score_name="loss")
    return {
        "defense": name,
        "test_accuracy": test_accuracy,
        "generalization_gap": generalization_gap,
        "auc": result.metrics["auc"],
        "tpr_at_fpr_0.01": result.metrics["tpr_at_fpr_0.01"],
        "tpr_at_fpr_0.001": result.metrics["tpr_at_fpr_0.001"],
        "extraction_fidelity": extraction_fidelity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip extraction measurements (much faster).")
    args = parser.parse_args()

    utils.setup_logging()
    config = utils.load_config(args.config)
    utils.set_seed(config["seed"])
    device = utils.get_device()
    results_dir = Path(config["results_dir"])

    pool = data.get_cifar10_pool(config["data"]["data_dir"])
    splits = data.make_splits(
        pool_size=len(pool),
        target_train_size=config["data"]["target_train_size"],
        target_holdout_size=config["data"]["target_holdout_size"],
        shadow_pool_size=config["data"]["shadow_pool_size"],
        attacker_aux_size=config["data"]["attacker_aux_size"],
        seed=config["seed"],
    )
    member_subset = data.get_subset(pool, splits.target_train)
    nonmember_subset = data.get_subset(pool, splits.target_holdout)
    batch_size = config["train"]["batch_size"]
    member_loader = DataLoader(member_subset, batch_size=batch_size, shuffle=True)
    nonmember_loader = DataLoader(nonmember_subset, batch_size=batch_size, shuffle=False)

    query_pool = None if args.skip_extraction else data.get_cifar100_query_pool(config["data"]["data_dir"])
    rows: list[dict] = []

    # ---------- Stage 1: cheap regularization defenses ----------
    undefended_artifacts = None
    for variant in config["defenses"]["cheap"]:
        name = variant["name"]
        utils.set_seed(config["seed"])
        logger.info("=== Cheap defense: %s ===", name)

        model = models.build_model(**{**config["model"], "dropout": variant.get("dropout", 0.0)})
        model, history = train.train_model(
            model, member_loader, nonmember_loader, device,
            epochs=variant.get("epochs", config["train"]["epochs"]),
            lr=config["train"]["lr"],
            weight_decay=variant.get("weight_decay", 0.0),
            label_smoothing=variant.get("label_smoothing", 0.0),
            early_stopping_patience=variant.get("early_stopping_patience"),
        )

        member_stats = train.compute_per_example_stats(model, member_subset, device)
        nonmember_stats = train.compute_per_example_stats(model, nonmember_subset, device)
        fidelity = None if args.skip_extraction else _measure_extraction_fidelity(
            model, nonmember_subset, query_pool, config, device, hardening={}
        )
        rows.append(_evaluate_defense(name, member_stats, nonmember_stats,
                                      history[-1]["eval_acc"], history[-1]["gap"], fidelity))

        if name == config["defenses"]["undefended_name"]:
            # Reused for Stage 2: output hardening needs no retraining, so it is
            # applied to this same undefended model.
            undefended_artifacts = (model, member_stats, nonmember_stats, history[-1])

    # ---------- Stage 2: output hardening (no retraining) ----------
    if undefended_artifacts is None:
        raise ValueError(
            f"No cheap-defense variant named '{config['defenses']['undefended_name']}' was found; "
            "output hardening needs that baseline model to apply to."
        )
    base_model, base_member_stats, base_nonmember_stats, base_history = undefended_artifacts

    for variant in config["defenses"]["hardening"]:
        name = variant["name"]
        logger.info("=== Output hardening: %s ===", name)
        hardening = {k: v for k, v in variant.items() if k != "name"}
        fidelity = None if args.skip_extraction else _measure_extraction_fidelity(
            base_model, nonmember_subset, query_pool, config, device, hardening=hardening
        )
        rows.append(_evaluate_defense(name, base_member_stats, base_nonmember_stats,
                                      base_history["eval_acc"], base_history["gap"],
                                      fidelity, hardening=hardening))

    # ---------- Stage 3: DP-SGD epsilon sweep ----------
    dp_rows = []
    target_delta = defenses.delta_for_dataset_size(len(splits.target_train))
    logger.info("DP delta fixed at 1/N = %.3g", target_delta)

    for epsilon in config["defenses"]["epsilons"]:
        epsilon_value = float("inf") if epsilon in ("inf", "infinity", None) else float(epsilon)
        utils.set_seed(config["seed"])
        logger.info("=== DP-SGD: epsilon=%s ===", epsilon_value)

        model = models.build_model(**config["model"])
        model, history, privacy_spent = train.train_model_dp(
            model, member_loader, nonmember_loader, device,
            epochs=config["defenses"]["dp_epochs"],
            lr=config["defenses"]["dp_lr"],
            target_epsilon=epsilon_value,
            target_delta=target_delta,
            max_grad_norm=config["defenses"]["max_grad_norm"],
        )

        member_stats = train.compute_per_example_stats(model, member_subset, device)
        nonmember_stats = train.compute_per_example_stats(model, nonmember_subset, device)
        fidelity = None if args.skip_extraction else _measure_extraction_fidelity(
            model, nonmember_subset, query_pool, config, device, hardening={}
        )

        label = "DP-SGD (eps=inf, no DP)" if not np.isfinite(epsilon_value) else f"DP-SGD (eps={epsilon_value:g})"
        row = _evaluate_defense(label, member_stats, nonmember_stats,
                                history[-1]["eval_acc"], history[-1]["gap"], fidelity)
        row["epsilon"] = epsilon_value
        row["achieved_epsilon"] = privacy_spent["epsilon"]
        rows.append(row)
        dp_rows.append(row)

    # ---------- Report ----------
    frame = pd.DataFrame(rows)
    table_path = results_dir / "tables" / "defense_comparison.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table_path, index=False)

    if dp_rows:
        dp_frame = pd.DataFrame(dp_rows).sort_values("epsilon")
        plots.plot_privacy_utility_tradeoff(
            dp_frame["epsilon"].tolist(),
            dp_frame["test_accuracy"].tolist(),
            dp_frame["auc"].tolist(),
            dp_frame["tpr_at_fpr_0.001"].tolist(),
            results_dir / "plots" / "privacy_utility_tradeoff.png",
        )

    plots.plot_defense_comparison(
        frame["defense"].tolist(),
        frame["test_accuracy"].tolist(),
        frame["tpr_at_fpr_0.001"].tolist(),
        results_dir / "plots" / "defense_comparison.png",
    )
    utils.save_json({"defenses": rows, "delta": target_delta},
                    results_dir / "metrics" / "defenses.json")

    print("\n=== Defense comparison ===")
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved table -> {table_path}")


if __name__ == "__main__":
    main()
