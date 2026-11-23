"""Phase 5: black-box model extraction against a trained target.

Sweeps the attacker's query budget and compares two API settings — full
probability vectors vs. label-only responses — reporting fidelity (agreement
with the target) and surrogate task accuracy at each budget.

Usage:
    python experiments/run_extraction.py --config configs/extraction.yaml

Requires a target checkpoint from experiments/train_targets.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, models, plots, utils
from src.attacks import extraction

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    utils.setup_logging()
    config = utils.load_config(args.config)
    utils.set_seed(config["seed"])
    device = utils.get_device()

    target_run = config["target_run_name"]
    results_dir = Path(config["results_dir"])

    # Load the victim model and hide it behind the black-box API.
    target_model = models.build_model(**config["model"])
    checkpoint_path = results_dir / "checkpoints" / f"{target_run}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No target checkpoint at {checkpoint_path}. Run experiments/train_targets.py first."
        )
    target_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    pool = data.get_cifar10_pool(config["data"]["data_dir"])
    splits = data.make_splits(
        pool_size=len(pool),
        target_train_size=config["data"]["target_train_size"],
        target_holdout_size=config["data"]["target_holdout_size"],
        shadow_pool_size=config["data"]["shadow_pool_size"],
        attacker_aux_size=config["data"]["attacker_aux_size"],
        seed=config["seed"],
    )
    # Fidelity is measured on data the target never trained on, so the number
    # reflects boundary agreement rather than shared memorization.
    eval_dataset = data.get_subset(pool, splits.target_holdout)
    query_pool = data.get_cifar100_query_pool(config["data"]["data_dir"])

    rows = []
    for output_mode, label_only in (("full", False), ("label", True)):
        for num_queries in config["extraction"]["query_budgets"]:
            if num_queries > len(query_pool):
                logger.warning("Skipping budget %d: attacker pool only has %d examples.",
                               num_queries, len(query_pool))
                continue

            api = extraction.BlackBoxAPI(
                target_model, device, output_mode=output_mode, query_budget=num_queries
            )
            stolen = extraction.collect_api_responses(api, query_pool, num_queries)
            assert api.query_count == num_queries, "API query accounting is wrong"

            utils.set_seed(config["seed"])
            surrogate = models.build_model(**config["extraction"]["surrogate_model"])
            surrogate = extraction.train_surrogate(
                surrogate,
                stolen,
                device,
                epochs=config["extraction"]["surrogate_epochs"],
                lr=config["extraction"]["surrogate_lr"],
                batch_size=config["train"]["batch_size"],
                label_only=label_only,
            )

            fidelity, surrogate_accuracy, target_accuracy = extraction.evaluate_extraction(
                surrogate, api, eval_dataset, device
            )
            rows.append(
                {
                    "output_mode": "probabilities" if output_mode == "full" else "label-only",
                    "num_queries": num_queries,
                    "fidelity": fidelity,
                    "surrogate_accuracy": surrogate_accuracy,
                    "target_accuracy": target_accuracy,
                }
            )
            logger.info("mode=%s queries=%d -> fidelity=%.4f surrogate_acc=%.4f",
                        output_mode, num_queries, fidelity, surrogate_accuracy)

    frame = pd.DataFrame(rows)
    table_path = results_dir / "tables" / f"{target_run}_extraction.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table_path, index=False)

    budgets = sorted(frame["num_queries"].unique())
    series = {}
    for mode_label in frame["output_mode"].unique():
        subset = frame[frame["output_mode"] == mode_label].sort_values("num_queries")
        series[f"fidelity ({mode_label})"] = subset["fidelity"].tolist()
        series[f"surrogate acc ({mode_label})"] = subset["surrogate_accuracy"].tolist()

    plots.plot_queries_vs_extraction(
        budgets, series, results_dir / "plots" / f"{target_run}_extraction.png",
        title=f"{target_run}: extraction cost vs quality",
    )
    utils.save_json({"target_run": target_run, "runs": rows},
                    results_dir / "metrics" / f"{target_run}_extraction.json")

    print(f"\n=== Extraction results for '{target_run}' ===")
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved table -> {table_path}")


if __name__ == "__main__":
    main()
