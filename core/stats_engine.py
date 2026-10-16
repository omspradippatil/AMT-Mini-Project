from __future__ import annotations

"""
core/stats_engine.py
────────────────────
Computes empirical and parametric statistics using SciPy.stats.

For a given array of scores and a chosen distribution family, this module:
  • Fits the SciPy distribution to the data (MLE).
  • Computes empirical moments via scipy.stats.describe().
  • Generates (x, pdf(x)) and (x, cdf(x)) arrays for plotting.
  • Returns a discrete PMF histogram (bin midpoints + relative frequencies).
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from dataclasses import dataclass, field


@dataclass
class StatsResult:
    # Empirical moments
    mean:     float = 0.0
    variance: float = 0.0
    std:      float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0   # excess kurtosis

    # Fitted distribution parameters
    dist_name:   str   = ""
    fit_params:  tuple = field(default_factory=tuple)

    # Curve data for plotting
    pdf_x: np.ndarray = field(default_factory=lambda: np.array([]))
    pdf_y: np.ndarray = field(default_factory=lambda: np.array([]))
    cdf_x: np.ndarray = field(default_factory=lambda: np.array([]))
    cdf_y: np.ndarray = field(default_factory=lambda: np.array([]))

    # Histogram data
    hist_bins:  np.ndarray = field(default_factory=lambda: np.array([]))
    hist_freq:  np.ndarray = field(default_factory=lambda: np.array([]))
    hist_edges: np.ndarray = field(default_factory=lambda: np.array([]))


# Map our dist names to scipy.stats distributions
_SCIPY_DIST_MAP: dict[str, object] = {
    "normal":      sp_stats.norm,
    "gamma":       sp_stats.gamma,
    "exponential": sp_stats.expon,
    "beta":        sp_stats.beta,
    "poisson":     sp_stats.norm,   # Poisson treated as Normal approx for KDE
}


def compute_empirical_stats(scores: np.ndarray, dist: str = "normal") -> StatsResult:
    """
    Fit a SciPy distribution to `scores` and compute all statistics.

    Parameters
    ----------
    scores : 1-D array of score values
    dist   : distribution family name

    Returns
    -------
    StatsResult dataclass
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]

    # ── Empirical moments via scipy.stats.describe ─────────────────────────
    desc = sp_stats.describe(scores)
    result = StatsResult(
        mean=float(desc.mean),
        variance=float(desc.variance),
        std=float(np.sqrt(desc.variance)),
        skewness=float(desc.skewness),
        kurtosis=float(desc.kurtosis),   # scipy returns excess kurtosis
        dist_name=dist,
    )

    # ── Fit SciPy distribution (MLE) ──────────────────────────────────────
    scipy_dist = _SCIPY_DIST_MAP.get(dist, sp_stats.norm)
    x_min = max(scores.min() - 5, 0.0)
    x_max = min(scores.max() + 5, 105.0)
    x_vals = np.linspace(x_min, x_max, 400)

    try:
        if dist == "normal" or dist == "poisson":
            fit_params = scipy_dist.fit(scores)
        elif dist == "gamma":
            fit_params = scipy_dist.fit(scores, floc=0)
        elif dist == "exponential":
            fit_params = scipy_dist.fit(scores, floc=0)
        elif dist == "beta":
            # Beta requires data in (0, 1); we scale to (0, 1)
            scaled = np.clip(scores / 100.0, 1e-6, 1 - 1e-6)
            fit_params = sp_stats.beta.fit(scaled, floc=0, fscale=1)
            result.fit_params = fit_params
            # PDF / CDF on original scale
            pdf_scaled = sp_stats.beta.pdf(x_vals / 100.0, *fit_params) / 100.0
            cdf_scaled = sp_stats.beta.cdf(x_vals / 100.0, *fit_params)
            result.pdf_x, result.pdf_y = x_vals, pdf_scaled
            result.cdf_x, result.cdf_y = x_vals, cdf_scaled
            result.hist_bins, result.hist_freq, result.hist_edges = _histogram(scores)
            return result
        else:
            fit_params = scipy_dist.fit(scores)

        result.fit_params = fit_params
        result.pdf_x = x_vals
        result.pdf_y = scipy_dist.pdf(x_vals, *fit_params)
        result.cdf_x = x_vals
        result.cdf_y = scipy_dist.cdf(x_vals, *fit_params)

    except Exception:
        # Fallback to KDE if fitting fails
        kde = sp_stats.gaussian_kde(scores)
        result.pdf_x = x_vals
        result.pdf_y = kde(x_vals)
        result.cdf_x = x_vals
        result.cdf_y = np.cumsum(result.pdf_y) * (x_vals[1] - x_vals[0])

    result.hist_bins, result.hist_freq, result.hist_edges = _histogram(scores)
    return result


def _histogram(scores: np.ndarray, n_bins: int = 20) -> tuple:
    """Return bin midpoints, relative frequencies, and edges."""
    counts, edges = np.histogram(scores, bins=n_bins, density=False)
    midpoints = (edges[:-1] + edges[1:]) / 2
    rel_freq = counts / counts.sum()
    return midpoints, rel_freq, edges


def parametric_cdf(score_val: float, dist: str, fit_params: tuple) -> float:
    """P(X ≤ score_val) using the fitted scipy distribution."""
    scipy_dist = _SCIPY_DIST_MAP.get(dist, sp_stats.norm)
    try:
        if dist == "beta":
            return float(sp_stats.beta.cdf(score_val / 100.0, *fit_params))
        return float(scipy_dist.cdf(score_val, *fit_params))
    except Exception:
        return float("nan")
