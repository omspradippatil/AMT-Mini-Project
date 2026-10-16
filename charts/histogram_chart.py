"""
charts/histogram_chart.py
─────────────────────────
Seaborn histogram with KDE overlay and score-cutoff marker.
Returns a matplotlib Figure for use with st.pyplot().
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(style="darkgrid")


def build_histogram(
    scores: np.ndarray,
    score_cutoff: float | None = None,
    dist_name: str = "Distribution",
    color: str = "#4C72B0",
    n_bins: int = 30,
) -> plt.Figure:
    """
    Seaborn histogram + KDE curve.

    Parameters
    ----------
    scores       : 1-D array of score values
    score_cutoff : optional vertical marker
    dist_name    : label for the chart title
    color        : primary bar color
    n_bins       : histogram bin count

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Histogram + KDE
    sns.histplot(
        scores,
        bins=n_bins,
        kde=True,
        color=color,
        alpha=0.65,
        edgecolor="white",
        linewidth=0.5,
        stat="density",
        ax=ax,
    )

    # Score cutoff vertical line
    if score_cutoff is not None:
        ax.axvline(
            score_cutoff,
            color="#E74C3C",
            linewidth=2.0,
            linestyle="--",
            label=f"Cutoff = {score_cutoff:.1f}",
        )
        ax.legend(fontsize=10)

    ax.set_title(f"Score Histogram + KDE  —  {dist_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(left=0)

    fig.tight_layout()
    return fig
