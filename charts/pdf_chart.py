"""
charts/pdf_chart.py
────────────────────
Matplotlib PDF curve with shaded probability tail area.
Returns a matplotlib Figure for use with st.pyplot().
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(style="darkgrid")


def build_pdf_chart(
    pdf_x: np.ndarray,
    pdf_y: np.ndarray,
    score_cutoff: float | None = None,
    direction: str = "above",
    dist_name: str = "Distribution",
    probability: float | None = None,
    color: str = "#2ECC71",
) -> plt.Figure:
    """
    PDF curve with shaded tail area indicating the computed probability.

    Parameters
    ----------
    pdf_x        : x-axis values (score range)
    pdf_y        : fitted PDF values
    score_cutoff : score threshold for tail shading
    direction    : "above" → shade right tail | "below" → shade left tail
    dist_name    : label for chart title
    probability  : computed probability value (shown in legend)
    color        : PDF line colour

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # PDF curve
    ax.plot(pdf_x, pdf_y, color=color, linewidth=2.5, label="PDF")
    ax.fill_between(pdf_x, pdf_y, alpha=0.15, color=color)

    # Shaded tail
    if score_cutoff is not None:
        if direction == "above":
            mask = pdf_x >= score_cutoff
            shade_label = (
                f"P(Score ≥ {score_cutoff:.1f}) = {probability:.3f}"
                if probability is not None
                else f"Score ≥ {score_cutoff:.1f}"
            )
        else:
            mask = pdf_x <= score_cutoff
            shade_label = (
                f"P(Score ≤ {score_cutoff:.1f}) = {probability:.3f}"
                if probability is not None
                else f"Score ≤ {score_cutoff:.1f}"
            )

        ax.fill_between(
            pdf_x,
            pdf_y,
            where=mask,
            alpha=0.45,
            color="#E74C3C",
            label=shade_label,
        )
        ax.axvline(
            score_cutoff,
            color="#E74C3C",
            linewidth=1.8,
            linestyle="--",
        )

    ax.set_title(f"Probability Density Function  —  {dist_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig
