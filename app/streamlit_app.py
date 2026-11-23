"""Privacy Report Card — a Streamlit dashboard over the lab's saved results.

Run with:
    streamlit run app/streamlit_app.py

This app is strictly a *reader*. It never trains a model or runs an attack:
it loads the CSV/JSON artifacts the experiment scripts already wrote. That
separation is deliberate — a dashboard that silently retrains things is a
dashboard whose numbers you can't trace back to a run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import utils

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# ---------------------------------------------------------------------------
# The risk rule. Documented here, in one place, because a "risk label" with an
# undocumented rule is worse than no label at all.
#
# The headline input is TPR at 0.1% FPR, not AUC. Remember the baseline: an
# attack with no signal at all scores TPR = 0.1% at 0.1% FPR. So:
#
#   HIGH    TPR@0.1%FPR >= 5%   (50x better than chance — the attacker can
#                                reliably single out ~1 in 20 training members)
#           or AUC >= 0.70      (large average-case leakage on its own)
#   MEDIUM  TPR@0.1%FPR >= 1%   (10x chance — a real, exploitable signal)
#           or AUC >= 0.60
#   LOW     otherwise           (NOT "safe" — see the caveat rendered below)
#
# These cut-offs are judgement calls for a teaching lab, not an industry
# standard. They are deliberately strict on the TPR axis and lenient on AUC,
# because AUC can look reassuringly near 0.5 on a model that still leaks a
# small number of examples catastrophically.
# ---------------------------------------------------------------------------
RISK_RULES = [
    ("HIGH", 0.05, 0.70),
    ("MEDIUM", 0.01, 0.60),
]

EXTRACTION_RULES = [("HIGH", 0.90), ("MEDIUM", 0.70)]


def membership_risk_label(auc: float, tpr_at_low_fpr: float) -> tuple[str, str]:
    """Return (label, explanation) for membership inference risk."""
    for label, tpr_threshold, auc_threshold in RISK_RULES:
        if tpr_at_low_fpr >= tpr_threshold:
            return label, (
                f"TPR@0.1%FPR of {tpr_at_low_fpr:.1%} is >= {tpr_threshold:.0%} "
                f"({tpr_at_low_fpr / 0.001:.0f}x the 0.1% a signal-free attack would get)."
            )
        if auc >= auc_threshold:
            return label, f"Attack AUC of {auc:.3f} is >= {auc_threshold:.2f}."
    return "LOW", (
        f"AUC {auc:.3f} and TPR@0.1%FPR {tpr_at_low_fpr:.2%} both fall below the "
        "MEDIUM cut-offs."
    )


def extraction_risk_label(fidelity: float) -> str:
    for label, threshold in EXTRACTION_RULES:
        if fidelity >= threshold:
            return label
    return "LOW"


def _render_risk_badge(label: str, explanation: str) -> None:
    renderer = {"HIGH": st.error, "MEDIUM": st.warning, "LOW": st.success}[label]
    renderer(f"**Membership inference risk: {label}** — {explanation}")


def _load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _show_plot(path: Path, caption: str) -> bool:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
        return True
    return False


def main() -> None:
    st.set_page_config(page_title="Privacy Report Card", page_icon="🔐", layout="wide")
    st.title("🔐 Model Privacy Report Card")
    st.caption(
        "Privacy risk assessment for models trained in this lab. All numbers are read "
        "from saved experiment artifacts in `results/` — nothing is computed here."
    )

    mia_tables = sorted(RESULTS_DIR.glob("tables/*_mia_comparison.csv"))
    if not mia_tables:
        st.warning(
            "No results found yet. Generate some with:\n\n"
            "```bash\n"
            "python experiments/train_targets.py --config configs/target_overfit.yaml\n"
            "python experiments/run_mia.py --config configs/mia_overfit.yaml\n"
            "```"
        )
        st.stop()

    run_names = [p.name.replace("_mia_comparison.csv", "") for p in mia_tables]
    selected_run = st.sidebar.selectbox("Target model", run_names)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Risk rule**\n\n"
        "- **HIGH**: TPR@0.1%FPR ≥ 5% or AUC ≥ 0.70\n"
        "- **MEDIUM**: TPR@0.1%FPR ≥ 1% or AUC ≥ 0.60\n"
        "- **LOW**: below both\n\n"
        "A signal-free attack scores TPR = 0.1% at 0.1% FPR, so 1% is already "
        "10× chance."
    )

    mia_table = _load_csv(RESULTS_DIR / "tables" / f"{selected_run}_mia_comparison.csv")
    training_metrics = None
    metrics_path = RESULTS_DIR / "metrics" / f"{selected_run}.json"
    if metrics_path.exists():
        training_metrics = utils.load_json(metrics_path)

    # The strongest attack is what defines the model's risk — a defender does
    # not get to be graded on the weakest attack anyone tried.
    tpr_column = "tpr_at_fpr_0.001"
    strongest = mia_table.loc[mia_table[tpr_column].idxmax()] if tpr_column in mia_table else None

    st.header(f"Report card: `{selected_run}`")

    columns = st.columns(5)
    if training_metrics:
        columns[0].metric("Test accuracy", f"{training_metrics['final_eval_acc']:.1%}")
        columns[1].metric("Generalization gap", f"{training_metrics['generalization_gap']:.1%}",
                          help="Train accuracy minus test accuracy. The engine of average-case leakage.")
    if strongest is not None:
        columns[2].metric("Best attack AUC", f"{strongest['auc']:.3f}",
                          help=f"Strongest attack: {strongest['attack']}")
        columns[3].metric("TPR @ 0.1% FPR", f"{strongest[tpr_column]:.2%}",
                          help="The headline privacy metric — worst-case leakage.")

    extraction_table = _load_csv(RESULTS_DIR / "tables" / f"{selected_run}_extraction.csv")
    best_fidelity = None
    if extraction_table is not None and not extraction_table.empty:
        best_fidelity = extraction_table["fidelity"].max()
        columns[4].metric("Extraction fidelity", f"{best_fidelity:.1%}",
                          help="Best agreement a stolen surrogate achieved with this model.")

    if strongest is not None:
        label, explanation = membership_risk_label(strongest["auc"], strongest[tpr_column])
        _render_risk_badge(label, explanation)
        if label == "LOW":
            st.caption(
                "⚠️ **LOW is not a guarantee.** These attacks are simplified and the "
                "shadow counts are small, so this is an empirical lower bound on "
                "leakage. Only DP-SGD provides an actual upper bound."
            )
    if best_fidelity is not None:
        st.info(
            f"**Model extraction risk: {extraction_risk_label(best_fidelity)}** — a surrogate "
            f"reached {best_fidelity:.1%} agreement with the target."
        )

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Attack comparison")
        if mia_table is not None:
            display_columns = [c for c in ("attack", "auc", "balanced_accuracy",
                                           "tpr_at_fpr_0.01", "tpr_at_fpr_0.001")
                               if c in mia_table.columns]
            st.dataframe(
                mia_table[display_columns].style.format(
                    {c: "{:.4f}" for c in display_columns if c != "attack"}
                ),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Attacks get stronger down this list. Watch the TPR@0.1%FPR column rather "
                "than AUC — that is where LiRA's per-example calibration shows up."
            )

    with right:
        st.subheader("ROC (log-log)")
        if not _show_plot(RESULTS_DIR / "plots" / f"{selected_run}_mia_roc.png",
                          "Low-FPR region is the part that matters"):
            st.caption("Not generated yet — run experiments/run_mia.py.")

    st.markdown("---")
    st.subheader("Score distributions")
    if not _show_plot(RESULTS_DIR / "plots" / f"{selected_run}_score_distributions.png",
                      "The attack works exactly to the extent these distributions separate"):
        st.caption("Not generated yet — run experiments/run_mia.py.")

    if extraction_table is not None:
        st.markdown("---")
        st.subheader("Model extraction")
        extraction_left, extraction_right = st.columns(2)
        with extraction_left:
            st.dataframe(extraction_table, use_container_width=True, hide_index=True)
        with extraction_right:
            _show_plot(RESULTS_DIR / "plots" / f"{selected_run}_extraction.png",
                       "Query budget vs surrogate quality")

    defense_table = _load_csv(RESULTS_DIR / "tables" / "defense_comparison.csv")
    if defense_table is not None:
        st.markdown("---")
        st.header("Defenses")
        st.caption(
            "Defenses are evaluated with the loss-threshold attack so every row is "
            "directly comparable. Note how the cheap defenses move AUC far more than "
            "they move TPR@0.1%FPR — average-case leakage is much easier to suppress "
            "than worst-case memorization."
        )
        st.dataframe(defense_table, use_container_width=True, hide_index=True)

        defense_left, defense_right = st.columns(2)
        with defense_left:
            _show_plot(RESULTS_DIR / "plots" / "privacy_utility_tradeoff.png",
                       "The core result: what privacy costs in accuracy")
        with defense_right:
            _show_plot(RESULTS_DIR / "plots" / "defense_comparison.png",
                       "Utility retained vs leakage removed")

    gap_table = _load_csv(RESULTS_DIR / "tables" / "gap_sweep_aggregated.csv")
    if gap_table is not None:
        st.markdown("---")
        st.header("Overfitting vs leakage")
        gap_left, gap_right = st.columns(2)
        with gap_left:
            _show_plot(RESULTS_DIR / "plots" / "gap_vs_attack_auc.png", "Gap vs attack AUC")
        with gap_right:
            _show_plot(RESULTS_DIR / "plots" / "gap_vs_attack_tpr.png", "Gap vs worst-case leakage")
        st.dataframe(gap_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        "Defensive research on self-trained models and public datasets only. "
        "See docs/threat_model.md for scope and README.md for limitations."
    )


if __name__ == "__main__":
    main()
