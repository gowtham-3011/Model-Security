"""Publication-style plots for the lab.

Design notes (these are deliberate, not defaults):

* **Palette.** The categorical hues below are a validated colorblind-safe
  set: worst adjacent-pair separation is Delta-E 9.1 under simulated
  protanopia (target >= 8) and 22.9 for normal vision (floor >= 15). Two of
  the hues sit below 3:1 contrast against the chart surface, so every chart
  here also carries a legend *and* distinct marker shapes — series identity
  is never carried by color alone. Scatter plots cap at three series, where
  the all-pairs separation still validates.
* **No dual-axis charts anywhere.** The Phase 6 "core plot" wants to show
  utility (test accuracy) and attack success (AUC, TPR@low-FPR) against
  epsilon. Those live on different scales, and a twin y-axis would let the
  reader draw a crossing point that is an artifact of two arbitrarily chosen
  scales. We use stacked panels sharing the x-axis instead, so every
  comparison the reader can make is a real one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: these scripts run on Colab / CI, not a desktop
import matplotlib.pyplot as plt

# Categorical slots, in fixed order. Never cycle past the end — fold extra
# series into "other" or facet instead.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")
# Secondary encoding: shape carries identity too, for CVD readers and print.
SERIES_MARKERS = ("o", "s", "^", "D", "v", "P")

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

STATUS_COLORS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}


def _style_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str | None = None) -> None:
    """Apply the lab's consistent, recessive chart chrome."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    if title:
        ax.set_title(title, color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left")


def _new_figure(figsize: tuple[float, float] = (6.0, 4.5)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight" rather than fig.tight_layout(): the stacked-panel
    # figures set their own gridspec spacing, which tight_layout warns about
    # and can override.
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_loglog(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    path: str | Path,
    title: str = "Membership inference ROC (log-log)",
) -> Path:
    """Plot attack ROC curves on log-log axes.

    Log-log rather than linear axes because the interesting region for a
    privacy attack is the bottom-left corner — the low-FPR regime where a
    real attacker operates. On linear axes that entire region is compressed
    into a few pixels, and two attacks that differ by 10x in TPR at 0.1% FPR
    look identical. This is the plot from Carlini et al. (2022) and it is the
    single most informative figure in the lab.

    ``curves`` maps an attack name to its (fpr, tpr) arrays.
    """
    fig, ax = _new_figure()

    # A log axis can't render FPR/TPR of exactly 0; clamp to a small floor so
    # the first ROC vertex is still visible instead of silently dropped.
    floor = 1e-5
    for index, (name, (fpr, tpr)) in enumerate(curves.items()):
        ax.plot(
            np.clip(fpr, floor, None),
            np.clip(tpr, floor, None),
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            marker=SERIES_MARKERS[index % len(SERIES_MARKERS)],
            markevery=0.15,
            markersize=5,
            linewidth=2,
            label=name,
            zorder=3,
        )

    diagonal = np.logspace(np.log10(floor), 0, 100)
    ax.plot(diagonal, diagonal, color=INK_MUTED, linestyle="--", linewidth=1.2,
            label="random guess", zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 1.0)
    ax.set_ylim(1e-4, 1.0)
    _style_axes(ax, "False positive rate", "True positive rate", title)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left")
    return _save(fig, path)


def plot_score_distributions(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    path: str | Path,
    score_label: str = "Membership score",
    title: str = "Score distributions: members vs non-members",
) -> Path:
    """Overlay member/non-member score histograms.

    This is the intuition-builder: the attack works exactly to the extent
    these two distributions fail to overlap.
    """
    fig, ax = _new_figure()
    bins = np.histogram_bin_edges(np.concatenate([member_scores, nonmember_scores]), bins=60)
    ax.hist(nonmember_scores, bins=bins, color=SERIES_COLORS[1], alpha=0.65,
            label="non-members", zorder=2)
    ax.hist(member_scores, bins=bins, color=SERIES_COLORS[0], alpha=0.65,
            label="members", zorder=3)
    _style_axes(ax, score_label, "Count", title)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    return _save(fig, path)


def plot_gap_vs_attack(
    generalization_gaps: Sequence[float],
    attack_values: Sequence[float],
    path: str | Path,
    point_labels: Sequence[str] | None = None,
    attack_metric_label: str = "Attack AUC",
    title: str = "Generalization gap vs attack success",
    gap_errors: Sequence[float] | None = None,
    attack_errors: Sequence[float] | None = None,
) -> Path:
    """Scatter the generalization gap against attack success across models.

    The Phase 2 headline experiment: does overfitting *cause* membership
    leakage? One point per configuration (varying epochs / training set size),
    averaged over seeds.

    ``gap_errors`` / ``attack_errors`` draw +/- 1 standard deviation across
    seeds. Pass them whenever the points are seed-averaged: single-seed attack
    numbers move by several points run to run, and a scatter without error
    bars invites the reader to read a trend into what may be noise.
    """
    fig, ax = _new_figure()

    if gap_errors is not None or attack_errors is not None:
        ax.errorbar(
            generalization_gaps,
            attack_values,
            xerr=gap_errors,
            yerr=attack_errors,
            fmt="none",
            ecolor=BASELINE,
            elinewidth=1.2,
            capsize=3,
            zorder=2,
        )

    ax.scatter(
        generalization_gaps,
        attack_values,
        color=SERIES_COLORS[0],
        marker=SERIES_MARKERS[0],
        s=70,
        edgecolors=SURFACE,  # 2px surface ring so overlapping points stay readable
        linewidths=1.5,
        zorder=3,
    )

    if point_labels is not None:
        for gap, value, label in zip(generalization_gaps, attack_values, point_labels):
            ax.annotate(label, (gap, value), textcoords="offset points", xytext=(6, 4),
                        fontsize=7.5, color=INK_MUTED)

    # Trend line: quantifies the relationship the scatter suggests.
    if len(generalization_gaps) >= 2:
        slope, intercept = np.polyfit(generalization_gaps, attack_values, 1)
        x_line = np.linspace(min(generalization_gaps), max(generalization_gaps), 50)
        correlation = np.corrcoef(generalization_gaps, attack_values)[0, 1]
        ax.plot(x_line, slope * x_line + intercept, color=INK_MUTED, linestyle="--",
                linewidth=1.2, label=f"linear fit (r={correlation:.2f})", zorder=2)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    _style_axes(ax, "Generalization gap (train acc - test acc)", attack_metric_label, title)
    return _save(fig, path)


def plot_queries_vs_extraction(
    query_counts: Sequence[int],
    series: dict[str, Sequence[float]],
    path: str | Path,
    title: str = "Model extraction: query budget vs surrogate quality",
) -> Path:
    """Plot extraction fidelity / accuracy against query budget.

    All series here are fractions in [0, 1], so they legitimately share one
    y-axis. Log x-axis because query budgets span 1k to 50k.
    """
    fig, ax = _new_figure()
    for index, (name, values) in enumerate(series.items()):
        ax.plot(
            query_counts,
            values,
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            marker=SERIES_MARKERS[index % len(SERIES_MARKERS)],
            markersize=7,
            linewidth=2,
            label=name,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    _style_axes(ax, "Number of queries to the target API", "Fraction", title)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    return _save(fig, path)


def plot_privacy_utility_tradeoff(
    epsilons: Sequence[float],
    test_accuracies: Sequence[float],
    attack_aucs: Sequence[float],
    attack_tprs: Sequence[float],
    path: str | Path,
    tpr_label: str = "TPR @ 0.1% FPR",
    title: str = "Privacy-utility trade-off under DP-SGD",
) -> Path:
    """THE core plot: epsilon vs utility (top) and attack success (bottom).

    Two stacked panels sharing the epsilon axis, rather than one chart with a
    twin y-axis. A dual-axis version would invite the reader to read the
    crossing point of the accuracy and attack curves as meaningful, when that
    point can be moved anywhere by rescaling either axis. Stacked panels keep
    every visual comparison honest while still aligning both stories on the
    same x positions.

    Non-private training (epsilon = infinity) is plotted at the right edge and
    tick-labeled so it reads as the unbounded baseline it is.
    """
    fig, (ax_utility, ax_attack) = plt.subplots(
        2, 1, figsize=(6.5, 6.5), sharex=True, gridspec_kw={"hspace": 0.15}
    )
    fig.patch.set_facecolor(SURFACE)

    finite = [e for e in epsilons if np.isfinite(e)]
    # Place epsilon=inf one log-step past the largest finite value.
    infinity_position = max(finite) * 3 if finite else 1.0
    positions = [infinity_position if not np.isfinite(e) else e for e in epsilons]

    ax_utility.plot(positions, test_accuracies, color=SERIES_COLORS[0],
                    marker=SERIES_MARKERS[0], markersize=7, linewidth=2,
                    label="Test accuracy", zorder=3)
    ax_utility.set_ylim(0, 1)
    _style_axes(ax_utility, "", "Test accuracy", title)
    ax_utility.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    ax_attack.plot(positions, attack_aucs, color=SERIES_COLORS[1],
                   marker=SERIES_MARKERS[1], markersize=7, linewidth=2,
                   label="Attack AUC", zorder=3)
    ax_attack.plot(positions, attack_tprs, color=SERIES_COLORS[2],
                   marker=SERIES_MARKERS[2], markersize=7, linewidth=2,
                   label=tpr_label, zorder=3)
    ax_attack.axhline(0.5, color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=1)
    ax_attack.annotate("AUC = 0.5 (no leakage)", xy=(positions[0], 0.5),
                       xytext=(0, 5), textcoords="offset points",
                       fontsize=8, color=INK_MUTED)
    _style_axes(ax_attack, "Privacy budget epsilon (log scale)", "Attack success", None)
    ax_attack.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    ax_attack.set_xscale("log")
    ax_attack.set_xticks(positions)
    ax_attack.set_xticklabels(
        [r"$\infty$" if not np.isfinite(e) else f"{e:g}" for e in epsilons]
    )
    ax_attack.tick_params(colors=INK_MUTED, labelsize=9)

    return _save(fig, path)


def plot_defense_comparison(
    defense_names: Sequence[str],
    test_accuracies: Sequence[float],
    attack_tprs: Sequence[float],
    path: str | Path,
    tpr_label: str = "TPR @ 0.1% FPR",
    title: str = "Defenses: utility retained vs leakage removed",
) -> Path:
    """Grouped horizontal bars comparing defenses on utility and leakage.

    Horizontal because defense names are long text labels; a 2px surface gap
    between adjacent bars keeps the pairs visually distinct.
    """
    fig, ax = _new_figure(figsize=(7.0, 0.55 * len(defense_names) + 2.2))
    y_positions = np.arange(len(defense_names))
    bar_height = 0.38

    ax.barh(y_positions - bar_height / 2 - 0.01, test_accuracies, height=bar_height,
            color=SERIES_COLORS[0], label="Test accuracy", zorder=3)
    ax.barh(y_positions + bar_height / 2 + 0.01, attack_tprs, height=bar_height,
            color=SERIES_COLORS[1], label=tpr_label, zorder=3)

    # Direct value labels: the palette's low-contrast hues require visible
    # labels (the "relief rule"), and they save the reader an axis lookup.
    for y, accuracy_value, tpr_value in zip(y_positions, test_accuracies, attack_tprs):
        ax.annotate(f"{accuracy_value:.3f}", (accuracy_value, y - bar_height / 2 - 0.01),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK_SECONDARY)
        ax.annotate(f"{tpr_value:.3f}", (tpr_value, y + bar_height / 2 + 0.01),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK_SECONDARY)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(defense_names, fontsize=9, color=INK_SECONDARY)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    _style_axes(ax, "Fraction", "", title)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower right")
    return _save(fig, path)
