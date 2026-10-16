"""
app.py  —  Probability Distribution Dashboard
══════════════════════════════════════════════
Interactive Streamlit web application that:

  • Visualises score probability distributions (histogram, PDF, CDF)
  • Performs symbolic MGF moment analysis via SymPy
  • Computes empirical statistics via SciPy
  • Predicts score probabilities based on completion-time thresholds

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Streamlit

# ── Local modules ─────────────────────────────────────────────────────────────
from core.data_generator import (
    generate_dataset, get_dist_display_name, get_default_params, DISTRIBUTIONS
)
from core.symbolic_mgf import compute_symbolic_mgf
from core.stats_engine import compute_empirical_stats
from core.predictor import predict_probability

from charts.histogram_chart import build_histogram
from charts.pdf_chart import build_pdf_chart
from charts.cdf_chart import build_cdf_chart, build_empirical_cdf_chart


# ═══════════════════════════════════════════════════════════════════════════════
# Page configuration
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Probability Distribution Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
        .main-title  { font-size: 2.1rem; font-weight: 700; color: #2C3E50; }
        .sub-title   { font-size: 1.0rem; color: #7F8C8D; margin-top: -10px; margin-bottom: 20px; }
        .metric-card { background: #F8F9FA; border-radius: 12px; padding: 16px; text-align: center; }
        .stat-label  { font-size: 0.82rem; color: #636E72; text-transform: uppercase; letter-spacing: 1px; }
        .stat-value  { font-size: 1.6rem; font-weight: 700; color: #2D3436; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Distribution colour palette
# ═══════════════════════════════════════════════════════════════════════════════

DIST_COLORS: dict[str, tuple[str, str, str]] = {
    "normal":      ("#4C72B0", "#2ECC71", "#9B59B6"),
    "gamma":       ("#E67E22", "#27AE60", "#8E44AD"),
    "exponential": ("#E74C3C", "#16A085", "#6C3483"),
    "beta":        ("#1ABC9C", "#2980B9", "#884EA0"),
    "poisson":     ("#F39C12", "#1A5276", "#6D4C41"),
}

DIST_PARAM_LABELS: dict[str, list[str]] = {
    "normal":      ["Mean (μ)", "Std Dev (σ)"],
    "gamma":       ["Shape (α)", "Scale (β)"],
    "exponential": ["Rate (λ)"],
    "beta":        ["Alpha (α)", "Beta (β)"],
    "poisson":     ["Lambda (λ)"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    """Render sidebar controls and return configuration dict."""

    st.sidebar.markdown("## ⚙️ Controls")
    st.sidebar.markdown("---")

    # Distribution selector
    dist = st.sidebar.selectbox(
        "📊 Distribution Family",
        options=DISTRIBUTIONS,
        format_func=get_dist_display_name,
        index=0,
    )

    # Dataset size
    n = st.sidebar.slider("👥 Dataset Size", min_value=100, max_value=2000, value=500, step=50)

    # Random seed
    seed = st.sidebar.number_input("🎲 Random Seed", min_value=0, max_value=9999, value=42, step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Distribution Parameters")

    # Dynamic parameter inputs
    defaults = get_default_params(dist)
    params: dict = {}

    if dist == "normal":
        params["mu"]    = st.sidebar.slider("Mean (μ)", 20.0, 95.0, float(defaults.get("mu", 70)), 1.0)
        params["sigma"] = st.sidebar.slider("Std Dev (σ)", 1.0, 25.0, float(defaults.get("sigma", 12)), 0.5)

    elif dist == "gamma":
        params["alpha"]      = st.sidebar.slider("Shape (α)", 1.0, 20.0, float(defaults.get("alpha", 8)), 0.5)
        params["beta_scale"] = st.sidebar.slider("Scale (β)", 1.0, 20.0, float(defaults.get("beta_scale", 9)), 0.5)

    elif dist == "exponential":
        rate = st.sidebar.slider("Rate (λ × 100)", 0.5, 5.0, float(defaults.get("lambda_", 1/60) * 100), 0.1)
        params["lambda_"] = rate / 100.0

    elif dist == "beta":
        params["alpha"]  = st.sidebar.slider("Alpha (α)", 0.5, 15.0, float(defaults.get("alpha", 5)), 0.5)
        params["beta_"]  = st.sidebar.slider("Beta (β)",  0.5, 15.0, float(defaults.get("beta_", 2)), 0.5)

    elif dist == "poisson":
        params["lambda_"] = st.sidebar.slider("Lambda (λ)", 10.0, 95.0, float(defaults.get("lambda_", 70)), 1.0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Prediction Controls")

    time_cutoff  = st.sidebar.slider("⏱️ Time Cutoff (min)", min_value=5, max_value=120, value=60, step=5)
    score_cutoff = st.sidebar.slider("📈 Score Cutoff", min_value=0, max_value=100, value=70, step=1)
    direction    = st.sidebar.radio("Direction", ["above", "below"], horizontal=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🖼️ Chart Options")
    n_bins = st.sidebar.slider("Histogram Bins", 10, 60, 30, 5)

    return {
        "dist":         dist,
        "n":            n,
        "seed":         int(seed),
        "params":       params,
        "time_cutoff":  time_cutoff,
        "score_cutoff": float(score_cutoff),
        "direction":    direction,
        "n_bins":       n_bins,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Distribution Explorer
# ═══════════════════════════════════════════════════════════════════════════════

def render_distribution_tab(cfg: dict, df: pd.DataFrame, stats_result, colors: tuple) -> None:
    col_hist, col_metrics = st.columns([3, 1])

    with col_hist:
        fig_hist = build_histogram(
            df["score"].to_numpy(),
            score_cutoff=cfg["score_cutoff"],
            dist_name=get_dist_display_name(cfg["dist"]),
            color=colors[0],
            n_bins=cfg["n_bins"],
        )
        st.pyplot(fig_hist, use_container_width=True)
        plt.close(fig_hist)

    with col_metrics:
        st.markdown("#### 📋 Empirical Moments")
        st.markdown(f"""
| Statistic | Value |
|---|---|
| **Mean** | `{stats_result.mean:.3f}` |
| **Variance** | `{stats_result.variance:.3f}` |
| **Std Dev** | `{stats_result.std:.3f}` |
| **Skewness** | `{stats_result.skewness:.3f}` |
| **Kurtosis** | `{stats_result.kurtosis:.3f}` |
| **Min** | `{df['score'].min():.2f}` |
| **Max** | `{df['score'].max():.2f}` |
""")

    st.markdown("---")
    col_pdf, col_cdf = st.columns(2)

    with col_pdf:
        fig_pdf = build_pdf_chart(
            stats_result.pdf_x,
            stats_result.pdf_y,
            score_cutoff=cfg["score_cutoff"],
            direction=cfg["direction"],
            dist_name=get_dist_display_name(cfg["dist"]),
            color=colors[1],
        )
        st.pyplot(fig_pdf, use_container_width=True)
        plt.close(fig_pdf)

    with col_cdf:
        fig_ecdf = build_empirical_cdf_chart(
            df["score"].to_numpy(),
            score_cutoff=cfg["score_cutoff"],
            dist_name=get_dist_display_name(cfg["dist"]),
        )
        st.pyplot(fig_ecdf, use_container_width=True)
        plt.close(fig_ecdf)

    st.markdown("---")
    with st.expander("📊 Raw Dataset Preview", expanded=False):
        st.dataframe(df.head(200), use_container_width=True)
        col_dl1, col_dl2 = st.columns(2)
        col_dl1.download_button(
            "⬇️ Download Full Dataset (CSV)",
            data=df.to_csv(index=False),
            file_name=f"scores_{cfg['dist']}.csv",
            mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Score Predictor
# ═══════════════════════════════════════════════════════════════════════════════

def render_predictor_tab(cfg: dict, df: pd.DataFrame, colors: tuple) -> None:
    pred = predict_probability(
        df,
        time_cutoff=cfg["time_cutoff"],
        score_cutoff=cfg["score_cutoff"],
        dist=cfg["dist"],
        direction=cfg["direction"],
    )

    # ── Summary metric cards ─────────────────────────────────────────────────
    st.markdown("### 🔮 Prediction Results")
    c1, c2, c3, c4 = st.columns(4)

    direction_sym = "≥" if cfg["direction"] == "above" else "≤"

    c1.metric(
        label=f"Empirical  P(score {direction_sym} {cfg['score_cutoff']:.0f})",
        value=f"{pred.empirical_prob:.3f}",
        delta=f"{pred.empirical_prob * 100:.1f}%",
    )
    c2.metric(
        label=f"Parametric P(score {direction_sym} {cfg['score_cutoff']:.0f})",
        value=f"{pred.parametric_prob:.3f}",
        delta=f"{pred.parametric_prob * 100:.1f}%",
    )
    c3.metric(
        label="Students (time ≤ cutoff)",
        value=f"{pred.sample_size}",
        delta=f"{pred.sample_size / pred.total_size * 100:.1f}% of total",
    )
    c4.metric(
        label="Total Dataset Size",
        value=f"{pred.total_size}",
    )

    st.info(
        f"**Interpretation**: Among **{pred.sample_size}** students who completed "
        f"within **{cfg['time_cutoff']} minutes**, the probability of a score "
        f"**{direction_sym} {cfg['score_cutoff']:.0f}** is "
        f"**{pred.empirical_prob:.1%}** (empirical) / "
        f"**{pred.parametric_prob:.1%}** (fitted {get_dist_display_name(cfg['dist'])}).",
        icon="📊",
    )

    st.markdown("---")

    # ── PDF with tail shading ────────────────────────────────────────────────
    if pred.stats is not None and pred.filtered_scores is not None and len(pred.filtered_scores) > 5:
        col_pdf2, col_cdf2 = st.columns(2)

        with col_pdf2:
            st.markdown("#### PDF of Filtered Scores (time ≤ cutoff)")
            fig_pdf2 = build_pdf_chart(
                pred.stats.pdf_x,
                pred.stats.pdf_y,
                score_cutoff=cfg["score_cutoff"],
                direction=cfg["direction"],
                dist_name=f"{get_dist_display_name(cfg['dist'])} | time ≤ {cfg['time_cutoff']}min",
                probability=pred.parametric_prob,
                color=colors[1],
            )
            st.pyplot(fig_pdf2, use_container_width=True)
            plt.close(fig_pdf2)

        with col_cdf2:
            st.markdown("#### CDF of Filtered Scores")
            cdf_prob = 1.0 - pred.parametric_prob if cfg["direction"] == "above" else pred.parametric_prob
            fig_cdf2 = build_cdf_chart(
                pred.stats.cdf_x,
                pred.stats.cdf_y,
                score_cutoff=cfg["score_cutoff"],
                probability=cdf_prob,
                dist_name=f"{get_dist_display_name(cfg['dist'])} | time ≤ {cfg['time_cutoff']}min",
                color=colors[2],
            )
            st.pyplot(fig_cdf2, use_container_width=True)
            plt.close(fig_cdf2)
    else:
        st.warning("⚠️ No students found within the selected time cutoff. Adjust the slider.")

    # ── Completion time distribution ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⏱️ Completion Time Distribution")
    fig_time, ax_time = plt.subplots(figsize=(9, 3.5))
    import seaborn as sns
    sns.histplot(
        df["completion_time_min"],
        bins=30, kde=True, color=colors[0], alpha=0.65,
        edgecolor="white", stat="density", ax=ax_time,
    )
    ax_time.axvline(cfg["time_cutoff"], color="#E74C3C", linewidth=2.0, linestyle="--",
                    label=f"Cutoff = {cfg['time_cutoff']} min")
    ax_time.set_xlabel("Completion Time (min)", fontsize=11)
    ax_time.set_ylabel("Density", fontsize=11)
    ax_time.set_title("Completion Time Histogram + KDE", fontsize=13, fontweight="bold")
    ax_time.legend()
    fig_time.tight_layout()
    st.pyplot(fig_time, use_container_width=True)
    plt.close(fig_time)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Moment Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def render_moment_tab(cfg: dict, stats_result, mgf_result: dict) -> None:
    st.markdown("### 🧮 Moment Generating Function (SymPy)")

    # ── MGF LaTeX display ────────────────────────────────────────────────────
    col_mgf1, col_mgf2 = st.columns(2)
    with col_mgf1:
        st.markdown("**Symbolic MGF** (generic parameters)")
        st.latex(mgf_result["mgf_latex"])
    with col_mgf2:
        st.markdown("**Parametric MGF** (substituted values)")
        st.latex(mgf_result["mgf_param_latex"])

    st.markdown("---")

    # ── Derivative steps ─────────────────────────────────────────────────────
    with st.expander("📐 Step-by-step Derivatives at t = 0", expanded=True):
        for k, deriv_str in enumerate(mgf_result["deriv_latex"], start=1):
            st.latex(deriv_str)

    st.markdown("---")

    # ── Symbolic vs Empirical moments comparison table ──────────────────────
    st.markdown("### 📊 Symbolic vs. Empirical Moments")

    raw = mgf_result["raw_moments"]
    e_x   = mgf_result["E_X"]
    var_x = mgf_result["Var_X"]
    std_x = var_x ** 0.5
    skew  = mgf_result["skewness"]
    kurt  = mgf_result["kurtosis"]

    comparison_df = pd.DataFrame(
        {
            "Statistic": ["E[X] (Mean)", "Var[X] (Variance)", "Std Dev",
                          "Skewness", "Excess Kurtosis",
                          "Raw Moment M₁", "Raw Moment M₂",
                          "Raw Moment M₃", "Raw Moment M₄"],
            "Symbolic (SymPy)": [
                f"{e_x:.4f}", f"{var_x:.4f}", f"{std_x:.4f}",
                f"{skew:.4f}", f"{kurt:.4f}",
                f"{raw[0]:.4f}", f"{raw[1]:.4f}", f"{raw[2]:.4f}", f"{raw[3]:.4f}",
            ],
            "Empirical (SciPy)": [
                f"{stats_result.mean:.4f}",
                f"{stats_result.variance:.4f}",
                f"{stats_result.std:.4f}",
                f"{stats_result.skewness:.4f}",
                f"{stats_result.kurtosis:.4f}",
                "—", "—", "—", "—",
            ],
            "Δ (Symbolic − Empirical)": [
                f"{e_x - stats_result.mean:+.4f}",
                f"{var_x - stats_result.variance:+.4f}",
                f"{std_x - stats_result.std:+.4f}",
                f"{skew - stats_result.skewness:+.4f}",
                f"{kurt - stats_result.kurtosis:+.4f}",
                "—", "—", "—", "—",
            ],
        }
    )
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Bar chart: symbolic vs empirical key moments ─────────────────────────
    st.markdown("### 📊 Moments Comparison Chart")
    labels   = ["Mean", "Std Dev", "Skewness", "Kurtosis"]
    sym_vals = [e_x, std_x, skew, kurt]
    emp_vals = [stats_result.mean, stats_result.std, stats_result.skewness, stats_result.kurtosis]

    x = np.arange(len(labels))
    width = 0.35

    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    bars1 = ax_bar.bar(x - width / 2, sym_vals, width, label="Symbolic (SymPy)",
                       color="#3498DB", alpha=0.85)
    bars2 = ax_bar.bar(x + width / 2, emp_vals, width, label="Empirical (SciPy)",
                       color="#E67E22", alpha=0.85)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=11)
    ax_bar.set_ylabel("Value", fontsize=11)
    ax_bar.set_title("Symbolic vs Empirical Moment Comparison", fontsize=13, fontweight="bold")
    ax_bar.legend(fontsize=10)
    ax_bar.axhline(0, color="black", linewidth=0.8)

    for bar in bars1:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, h + abs(h) * 0.03,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="#2980B9")
    for bar in bars2:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, h + abs(h) * 0.03,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="#E74C3C")

    fig_bar.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)

    # ── Raw moment breakdown ─────────────────────────────────────────────────
    with st.expander("📐 Raw Moment Values (M₁ – M₄)", expanded=False):
        fig_raw, ax_raw = plt.subplots(figsize=(7, 3.5))
        ax_raw.bar(
            [f"M₁\n(E[X])", f"M₂\n(E[X²])", f"M₃\n(E[X³])", f"M₄\n(E[X⁴])"],
            raw,
            color=["#3498DB", "#2ECC71", "#E67E22", "#9B59B6"],
            alpha=0.85,
            edgecolor="white",
        )
        ax_raw.set_title("Raw Moments M₁ – M₄  (Symbolic)", fontsize=12, fontweight="bold")
        ax_raw.set_ylabel("Value")
        for i, v in enumerate(raw):
            ax_raw.text(i, v * 1.01, f"{v:.2e}", ha="center", va="bottom", fontsize=9)
        fig_raw.tight_layout()
        st.pyplot(fig_raw, use_container_width=True)
        plt.close(fig_raw)


# ═══════════════════════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        '<p class="main-title">🎯 Probability Distribution Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-title">Interactive visualisation of score distributions, '
        'MGF moment analysis, and completion-time threshold predictions</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    cfg = render_sidebar()
    colors = DIST_COLORS.get(cfg["dist"], ("#4C72B0", "#2ECC71", "#9B59B6"))

    # ── Data generation (cached) ─────────────────────────────────────────────
    @st.cache_data(show_spinner="Generating dataset…")
    def get_data(dist, n, params_tuple, seed):
        params = dict(params_tuple)
        return generate_dataset(n=n, dist=dist, params=params, seed=seed)

    df = get_data(
        cfg["dist"],
        cfg["n"],
        tuple(sorted(cfg["params"].items())),
        cfg["seed"],
    )

    # ── Statistics (cached) ──────────────────────────────────────────────────
    @st.cache_data(show_spinner="Computing statistics…")
    def get_stats(scores_tuple, dist):
        return compute_empirical_stats(np.array(scores_tuple), dist)

    stats_result = get_stats(tuple(df["score"].tolist()), cfg["dist"])

    # ── MGF / symbolic moments (cached) ─────────────────────────────────────
    @st.cache_data(show_spinner="Deriving MGF via SymPy…")
    def get_mgf(dist, params_tuple):
        params = dict(params_tuple)
        return compute_symbolic_mgf(dist, params)

    with st.spinner("Computing symbolic moments via SymPy…"):
        mgf_result = get_mgf(cfg["dist"], tuple(sorted(cfg["params"].items())))

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📊 Distribution Explorer",
        "🔮 Score Predictor",
        "🧮 Moment Analysis",
    ])

    with tab1:
        render_distribution_tab(cfg, df, stats_result, colors)

    with tab2:
        render_predictor_tab(cfg, df, colors)

    with tab3:
        render_moment_tab(cfg, stats_result, mgf_result)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<small>Built with **Streamlit** · **SymPy** · **SciPy** · **Matplotlib** · **Seaborn** · **NumPy** · **Pandas**</small>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
