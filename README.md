# Probability Distribution Dashboard

An interactive Streamlit web dashboard for visualising score probability distributions and predicting probabilities based on completion-time thresholds.

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
