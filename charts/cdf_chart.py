"""
charts/cdf_chart.py
────────────────────
Matplotlib CDF curve with reference lines for score cutoff and probability.
Returns a matplotlib Figure for use with st.pyplot().
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


def build_cdf_chart(
    cdf_x: np.ndarray,
    cdf_y: np.ndarray,
    score_cutoff: float | None = None,
    probability: float | None = None,
    dist_name: str = "Distribution",
    color: str = "#9B59B6",
) -> plt.Figure:
    """
    CDF curve with horizontal and vertical reference lines.

    Parameters
    ----------
    cdf_x        : x-axis values (score range)
    cdf_y        : CDF values ∈ [0, 1]
    score_cutoff : draws a vertical dashed line at this score
    probability  : draws a horizontal dashed line at this probability
    dist_name    : label for chart title
    color        : CDF line colour

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(cdf_x, cdf_y, color=color, linewidth=2.5, label="CDF")

    if score_cutoff is not None:
        ax.axvline(
            score_cutoff,
            color="#E74C3C",
            linewidth=1.8,
            linestyle="--",
            label=f"Score = {score_cutoff:.1f}",
        )

    if probability is not None:
        ax.axhline(
            probability,
            color="#E67E22",
            linewidth=1.8,
            linestyle=":",
            label=f"P = {probability:.3f}",
        )

        # Intersection dot
        if score_cutoff is not None:
            ax.scatter(
                [score_cutoff],
                [probability],
                s=80,
                zorder=5,
                color="#E74C3C",
            )

    ax.set_title(
        f"Cumulative Distribution Function  —  {dist_name}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Score", fontsize=11)
    ax.set_ylabel("Cumulative Probability", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.legend(fontsize=10)

    fig.tight_layout()
    return fig


def build_empirical_cdf_chart(
    scores: np.ndarray,
    score_cutoff: float | None = None,
    dist_name: str = "Distribution",
) -> plt.Figure:
    """
    Empirical (step) CDF from raw score data using seaborn ecdfplot.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    sns.ecdfplot(scores, ax=ax, color="#9B59B6", linewidth=2.5, label="Empirical CDF")

    if score_cutoff is not None:
        ecdf_val = float(np.mean(scores <= score_cutoff))
        ax.axvline(score_cutoff, color="#E74C3C", linewidth=1.8, linestyle="--",
                   label=f"Score = {score_cutoff:.1f}")
        ax.axhline(ecdf_val, color="#E67E22", linewidth=1.8, linestyle=":",
                   label=f"F({score_cutoff:.1f}) = {ecdf_val:.3f}")
        ax.scatter([score_cutoff], [ecdf_val], s=80, zorder=5, color="#E74C3C")

    ax.set_title(f"Empirical CDF  —  {dist_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Score", fontsize=11)
    ax.set_ylabel("Cumulative Probability", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)

    fig.tight_layout()
    return fig
