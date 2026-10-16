"""
core/symbolic_mgf.py
────────────────────
Uses SymPy to:
  1. Define the analytical Moment Generating Function M_X(t) for each
     supported distribution.
  2. Derive moments E[X^k] = M^(k)(0)  (kth derivative at t = 0).
  3. Compute central moments, variance, skewness, and excess kurtosis.
  4. Return LaTeX strings for display in Streamlit.

Supported distributions:
  "normal"      – M(t) = exp(μt + ½σ²t²)
  "gamma"       – M(t) = (1 − βt)^(−α),  t < 1/β
  "exponential" – M(t) = λ/(λ − t),       t < λ
  "beta"        – M(t) = 1F1(α; α+β; t)  (Kummer confluent hypergeometric)
                  → moments computed via ratio formula, MGF shown symbolically
  "poisson"     – M(t) = exp(λ(e^t − 1))
"""

from __future__ import annotations

import sympy as sp
from sympy import (
    Symbol, exp, diff, simplify, latex, oo,
    Rational, factorial, hyper, Abs, limit
)


# ── Shared symbols ────────────────────────────────────────────────────────────
t = Symbol("t", real=True)
mu_sym   = Symbol("mu",    positive=True)
sigma_sym = Symbol("sigma", positive=True)
alpha_sym = Symbol("alpha", positive=True)
beta_sym  = Symbol("beta",  positive=True)
lam_sym   = Symbol("lambda", positive=True)


# ── MGF expressions (symbolic, with symbolic parameters) ──────────────────────

def _normal_mgf() -> sp.Expr:
    return exp(mu_sym * t + Rational(1, 2) * sigma_sym**2 * t**2)

def _gamma_mgf() -> sp.Expr:
    # (1 − β·t)^(−α), valid for t < 1/β
    return (1 - beta_sym * t) ** (-alpha_sym)

def _exponential_mgf() -> sp.Expr:
    # λ / (λ − t)
    return lam_sym / (lam_sym - t)

def _poisson_mgf() -> sp.Expr:
    return exp(lam_sym * (exp(t) - 1))

def _beta_mgf() -> sp.Expr:
    """
    Kummer confluent hypergeometric 1F1(α; α+β; 100t).
    Scores are X = 100*Y where Y ~ Beta(α, β).
    """
    return hyper([alpha_sym], [alpha_sym + beta_sym], 100 * t)


_MGF_MAP = {
    "normal":      _normal_mgf,
    "gamma":       _gamma_mgf,
    "exponential": _exponential_mgf,
    "beta":        _beta_mgf,
    "poisson":     _poisson_mgf,
}

_PARAM_SUBS: dict[str, dict] = {}  # populated in compute_symbolic_mgf


def _substitute(expr: sp.Expr, dist: str, params: dict) -> sp.Expr:
    """Replace symbolic parameters with numeric values."""
    subs: dict[sp.Symbol, float] = {}
    if dist == "normal":
        subs = {mu_sym: params.get("mu", 70), sigma_sym: params.get("sigma", 12)}
    elif dist == "gamma":
        subs = {alpha_sym: params.get("alpha", 8), beta_sym: params.get("beta_scale", 9)}
    elif dist == "exponential":
        subs = {lam_sym: params.get("lambda_", 1 / 60)}
    elif dist == "beta":
        subs = {alpha_sym: params.get("alpha", 5), beta_sym: params.get("beta_", 2)}
    elif dist == "poisson":
        subs = {lam_sym: params.get("lambda_", 70)}
    return expr.subs(subs)


# ── Moment computation helpers ────────────────────────────────────────────────

def _raw_moments_from_mgf(mgf_numeric: sp.Expr, order: int = 4) -> list[float]:
    """
    Compute raw moments E[X^k] = M^(k)(0) for k = 1..order
    using symbolic differentiation at t = 0.
    Falls back to limit() if direct substitution fails.
    """
    moments = []
    expr = mgf_numeric
    for k in range(1, order + 1):
        expr = sp.diff(expr, t)
        try:
            val = float(expr.subs(t, 0))
        except Exception:
            val = float(sp.limit(expr, t, 0))
        moments.append(val)
    return moments


def _beta_moments_direct(alpha: float, beta_: float, order: int = 4) -> list[float]:
    """
    Beta distribution raw moments via:
    E[X^k] = prod_{j=0}^{k-1} (α+j) / (α+β+j)
    Scores are Beta * 100, so E[(100X)^k] = 100^k * E[X^k].
    """
    moments = []
    for k in range(1, order + 1):
        num = 1.0
        den = 1.0
        for j in range(k):
            num *= (alpha + j)
            den *= (alpha + beta_ + j)
        moments.append((num / den) * (100 ** k))
    return moments


# ── Main public function ──────────────────────────────────────────────────────

def compute_symbolic_mgf(dist: str, params: dict) -> dict:
    """
    Compute MGF, LaTeX string, and the first four moments symbolically.

    Parameters
    ----------
    dist   : one of "normal" | "gamma" | "exponential" | "beta" | "poisson"
    params : distribution parameter dict (see data_generator.DEFAULT_PARAMS)

    Returns
    -------
    dict with keys:
        mgf_latex   – LaTeX string of M_X(t)
        mgf_param_latex – M_X(t) with parameters substituted
        E_X         – float  (mean)
        Var_X       – float  (variance)
        skewness    – float  (standardised 3rd central moment)
        kurtosis    – float  (excess kurtosis = μ₄/σ⁴ − 3)
        raw_moments – list[float]  [M1, M2, M3, M4]
        deriv_latex – list[str]  LaTeX for M'(0), M''(0), M'''(0), M''''(0)
    """
    if dist not in _MGF_MAP:
        raise ValueError(f"Unsupported distribution: {dist!r}")

    # 1. Symbolic MGF (generic parameters)
    mgf_sym = _MGF_MAP[dist]()
    mgf_latex_str = f"M_X(t) = {latex(mgf_sym)}"

    # 2. Numeric MGF (parameters substituted)
    mgf_num = _substitute(mgf_sym, dist, params)
    if dist == "beta":
        # Sympy hyperexpand has a bug with float shift parameters (range(r-1) on Float)
        mgf_param_latex_str = f"M_X(t) = {latex(mgf_num)}"
    elif dist == "poisson":
        lam_val = params.get("lambda_", 70.0)
        mgf_param_latex_str = f"M_X(t) = e^{{{lam_val:.1f}(e^t - 1)}}"
    else:
        try:
            mgf_param_latex_str = f"M_X(t) = {latex(simplify(mgf_num))}"
        except Exception:
            mgf_param_latex_str = f"M_X(t) = {latex(mgf_num)}"

    # 3. Raw moments
    if dist == "beta":
        # Use direct formula to avoid hyper-function derivative complexity
        raw = _beta_moments_direct(params.get("alpha", 5), params.get("beta_", 2))
    else:
        raw = _raw_moments_from_mgf(mgf_num)

    m1, m2, m3, m4 = raw

    # 4. Central moments
    var = m2 - m1 ** 2                             # σ²
    mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3          # 3rd central moment
    mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4  # 4th

    std = var ** 0.5 if var > 0 else 1e-9
    skew = mu3 / (std ** 3) if std > 0 else 0.0
    kurt = (mu4 / (std ** 4) - 3) if std > 0 else 0.0

    # 5. LaTeX strings for each derivative evaluated at 0
    deriv_latex = []
    if dist == "beta":
        for k in range(1, 5):
            deriv_latex.append(
                f"M_X^{{({k})}}(0) = {raw[k-1]:.4f}"
            )
    else:
        expr = mgf_num
        for k in range(1, 5):
            expr = sp.diff(expr, t)
            sub_val = expr.subs(t, 0)
            try:
                sub_val = sp.simplify(sub_val)
            except Exception:
                pass
            deriv_latex.append(
                f"M_X^{{({k})}}(0) = {latex(sub_val)}"
            )

    return {
        "mgf_latex":       mgf_latex_str,
        "mgf_param_latex": mgf_param_latex_str,
        "E_X":             float(m1),
        "Var_X":           float(var),
        "skewness":        float(skew),
        "kurtosis":        float(kurt),
        "raw_moments":     [float(x) for x in raw],
        "deriv_latex":     deriv_latex,
    }
