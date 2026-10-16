from __future__ import annotations

"""
core/data_generator.py
─────────────────────
Generates a synthetic student dataset with columns:
  student_id, score (0–100), completion_time_min

Supported distribution families:
  "normal"      – scores ~ N(μ, σ²), times ~ N(45, 10²)
  "gamma"       – scores ~ Gamma(a=8, scale=9), times ~ Gamma(a=5, scale=9)
  "exponential" – scores ~ Exp(scale=60), times ~ Exp(scale=40)
  "beta"        – scores ~ Beta(α=5, β=2) * 100, times ~ N(50, 12²)
  "poisson"     – scores ~ Poisson(λ=70), times ~ N(55, 15²)
"""

import numpy as np
import pandas as pd
from scipy import stats


DISTRIBUTIONS = ["normal", "gamma", "exponential", "beta", "poisson"]

# Default parameters for each distribution
DEFAULT_PARAMS: dict[str, dict] = {
    "normal":      {"mu": 70.0,  "sigma": 12.0},
    "gamma":       {"alpha": 8.0, "beta_scale": 9.0},
    "exponential": {"lambda_": 1 / 60},
    "beta":        {"alpha": 5.0, "beta_": 2.0},
    "poisson":     {"lambda_": 70.0},
}

# Parameters for completion time (always Normal for simplicity)
TIME_PARAMS: dict[str, dict] = {
    "normal":      {"mu": 45, "sigma": 10},
    "gamma":       {"mu": 50, "sigma": 12},
    "exponential": {"mu": 40, "sigma": 10},
    "beta":        {"mu": 50, "sigma": 12},
    "poisson":     {"mu": 55, "sigma": 15},
}


def generate_dataset(
    n: int = 500,
    dist: str = "normal",
    params: dict | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic DataFrame of student scores and completion times.

    Parameters
    ----------
    n    : number of students
    dist : one of DISTRIBUTIONS
    params : optional param overrides (keys depend on distribution)
    seed : random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns [student_id, score, completion_time_min]
    """
    rng = np.random.default_rng(seed)
    p = {**DEFAULT_PARAMS.get(dist, {}), **(params or {})}
    tp = TIME_PARAMS.get(dist, {"mu": 45, "sigma": 10})

    # --- Score generation ---
    if dist == "normal":
        raw_scores = rng.normal(p["mu"], p["sigma"], n)
    elif dist == "gamma":
        raw_scores = rng.gamma(shape=p["alpha"], scale=p["beta_scale"], size=n)
    elif dist == "exponential":
        raw_scores = rng.exponential(scale=1.0 / p["lambda_"], size=n)
    elif dist == "beta":
        raw_scores = rng.beta(p["alpha"], p["beta_"], n) * 100.0
    elif dist == "poisson":
        raw_scores = rng.poisson(lam=p["lambda_"], size=n).astype(float)
    else:
        raise ValueError(f"Unknown distribution: {dist!r}. Choose from {DISTRIBUTIONS}")

    # Clip scores to [0, 100]
    scores = np.clip(raw_scores, 0.0, 100.0)

    # --- Completion time generation (Normal for all) ---
    times = rng.normal(tp["mu"], tp["sigma"], n)
    times = np.clip(times, 1.0, 120.0)

    return pd.DataFrame(
        {
            "student_id": np.arange(1, n + 1),
            "score": np.round(scores, 2),
            "completion_time_min": np.round(times, 2),
        }
    )


def get_dist_display_name(dist: str) -> str:
    """Human-readable name for each distribution."""
    names = {
        "normal":      "Normal (Gaussian)",
        "gamma":       "Gamma",
        "exponential": "Exponential",
        "beta":        "Beta",
        "poisson":     "Poisson",
    }
    return names.get(dist, dist.capitalize())


def get_default_params(dist: str) -> dict:
    """Return the default parameter dict for a given distribution."""
    return dict(DEFAULT_PARAMS.get(dist, {}))
