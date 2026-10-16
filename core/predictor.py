from __future__ import annotations

"""
core/predictor.py
─────────────────
Predicts score probabilities given completion-time and score thresholds.

Logic:
  1. Filter dataset to rows where completion_time_min ≤ time_cutoff.
  2. Among filtered rows, compute:
     a. Empirical probability: count(score OP score_cutoff) / total
     b. Parametric probability: from fitted SciPy CDF
  3. Return all values + sample size for display.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from core.stats_engine import compute_empirical_stats, parametric_cdf, StatsResult


@dataclass
class PredictionResult:
    time_cutoff:    float = 0.0
    score_cutoff:   float = 0.0
    direction:      str   = "above"         # "above" or "below"
    sample_size:    int   = 0               # rows matching time filter
    total_size:     int   = 0               # total rows in dataset
    empirical_prob: float = 0.0             # from filtered subset counts
    parametric_prob: float = 0.0            # from fitted CDF
    filtered_scores: np.ndarray = None      # scores in filtered subset
    stats: StatsResult = None               # stats on filtered scores


def predict_probability(
    df: pd.DataFrame,
    time_cutoff: float,
    score_cutoff: float,
    dist: str = "normal",
    direction: str = "above",
) -> PredictionResult:
    """
    Predict P(score OP score_cutoff | completion_time ≤ time_cutoff).

    Parameters
    ----------
    df           : DataFrame with columns [score, completion_time_min]
    time_cutoff  : maximum allowed completion time (minutes)
    score_cutoff : the score threshold
    dist         : distribution family name (for parametric CDF fitting)
    direction    : "above"  → P(score ≥ score_cutoff)
                   "below"  → P(score ≤ score_cutoff)

    Returns
    -------
    PredictionResult dataclass
    """
    result = PredictionResult(
        time_cutoff=time_cutoff,
        score_cutoff=score_cutoff,
        direction=direction,
        total_size=len(df),
    )

    # ── Filter by time ───────────────────────────────────────────────────────
    filtered = df[df["completion_time_min"] <= time_cutoff]
    result.sample_size = len(filtered)

    if result.sample_size == 0:
        result.filtered_scores = np.array([])
        result.empirical_prob  = 0.0
        result.parametric_prob = 0.0
        return result

    scores = filtered["score"].to_numpy(dtype=float)
    result.filtered_scores = scores

    # ── Empirical probability ────────────────────────────────────────────────
    if direction == "above":
        emp_count = np.sum(scores >= score_cutoff)
    else:
        emp_count = np.sum(scores <= score_cutoff)
    result.empirical_prob = float(emp_count / result.sample_size)

    # ── Parametric probability from fitted SciPy CDF ───────────────────────
    try:
        stats_result = compute_empirical_stats(scores, dist)
        result.stats = stats_result
        p_cdf = parametric_cdf(score_cutoff, dist, stats_result.fit_params)
        if direction == "above":
            result.parametric_prob = float(1.0 - p_cdf) if not np.isnan(p_cdf) else 0.0
        else:
            result.parametric_prob = float(p_cdf) if not np.isnan(p_cdf) else 0.0
    except Exception:
        result.parametric_prob = result.empirical_prob

    return result
