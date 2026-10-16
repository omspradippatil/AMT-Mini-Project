# Probability Distribution Dashboard

An interactive Streamlit web dashboard for visualising score probability distributions and predicting probabilities based on completion-time thresholds.

[![Deploy to GitHub Pages](https://github.com/omspradippatil/AMT-Mini-Project/actions/workflows/deploy.yml/badge.svg)](https://github.com/omspradippatil/AMT-Mini-Project/actions/workflows/deploy.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-2ea44f?style=for-the-badge&logo=github)](https://omspradippatil.github.io/AMT-Mini-Project/)

🚀 **Live Deployment URL:** [https://omspradippatil.github.io/AMT-Mini-Project/](https://omspradippatil.github.io/AMT-Mini-Project/)

---

## Features

| Component | Description |
|---|---|
| **Distribution Explorer** | Histogram + KDE, PDF curve, Empirical CDF |
| **Score Predictor** | P(score ≥/≤ cutoff \| time ≤ cutoff) — empirical + parametric |
| **Moment Analysis** | SymPy MGF (LaTeX), step-by-step derivatives, symbolic vs empirical moments table |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Supported Distributions

| Family | MGF M(t) |
|---|---|
| Normal | exp(μt + ½σ²t²) |
| Gamma | (1 − βt)^(−α) |
| Exponential | λ/(λ − t) |
| Beta | ₁F₁(α; α+β; t) |
| Poisson | exp(λ(eᵗ − 1)) |

## Project Structure

```
AMT Project/
├── app.py                   # Streamlit entrypoint
├── requirements.txt
├── core/
│   ├── data_generator.py    # Synthetic dataset (NumPy)
│   ├── symbolic_mgf.py      # SymPy MGF derivations
│   ├── stats_engine.py      # SciPy empirical stats
│   └── predictor.py         # Time-threshold probability predictor
└── charts/
    ├── histogram_chart.py   # Seaborn histogram + KDE
    ├── pdf_chart.py         # Matplotlib PDF + tail shading
    └── cdf_chart.py         # Matplotlib CDF + empirical ECDF
```
