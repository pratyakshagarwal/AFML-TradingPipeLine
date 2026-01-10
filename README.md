# AFML Trading Pipeline (v1)

## Overview

This repository implements a **research-grade algorithmic trading pipeline** grounded in the methodology of *Advances in Financial Machine Learning* (Marcos López de Prado).

The objective of this project is **not** to present a production-ready or consistently profitable strategy, but to build a **realistic, leakage-aware experimental framework** for short-horizon alpha research using high-frequency crypto market data.

Version 1 prioritizes:
- Correctness over optimization  
- Causality over convenience  
- Robust evaluation over headline performance  

The pipeline is fully modular and designed to expose fragile signals rather than mask them.

---

## Problem Statement

The core problem addressed is:

> **How to generate, evaluate, and backtest ML-based trading signals under realistic market constraints while strictly preventing information leakage.**

The pipeline:
1. Generates signals using AFML-compliant labeling and sampling techniques  
2. Trains ML models on non-uniform, event-driven data  
3. Evaluates them using purged, embargoed cross-validation  
4. Simulates execution with costs, slippage, and barrier-based exits  

---

## Data

- **Asset:** BTC (Binance)
- **Source:** Binance real-time trade API
- **Sampling:** 100 ms trade data
- **Length:** ~500,000 rows (~5 hours of market activity)

### Data Fetching

A custom `DataFetcher` is implemented with:
- Asynchronous I/O
- Runtime streaming and persistence
- Configurable:
  - asset
  - interval (ms)
  - trade type (`trades` / `order book`)

Only **trade data** is used in v1.

---

## Feature Engineering

### Resampling

- Raw trades are resampled into **volume bars**
- Eliminates time-based distortions and heteroskedasticity

### Features

Generated on volume bars:

- Returns
- Rolling return mean & standard deviation
- Z-score
- EMA (fast / slow)
- Return sign
- Cumulative return sign
- Lagged features

All features are **strictly causal**.

### Labeling

- **Triple-Barrier Labeling**
  - Upper barrier (profit-taking)
  - Lower barrier (stop-loss)
  - Time-based barrier
- Barrier widths scale with estimated volatility

### Sample Weighting

A custom `SampleWeightGenerator` computes weights based on:
- **Time decay**
- **Label uniqueness**

### Sequential Bootstrapping

- Used to select the most informative and least redundant samples
- Implemented with **Numba** for computational efficiency

### Stationarity

- **Fractional Differencing** applied to selected features
- Preserves memory while enforcing stationarity

---

## Modeling

- **Model:** `RandomForestClassifier`
- **Train/Test Split:** Event-aligned
- Focus is on:
  - Stability
  - Robustness
  - Interpretability

No hyperparameter overfitting or aggressive tuning in v1.

---

## Evaluation

### Cross-Validation

- **Purged K-Fold Cross-Validation**
- Overlapping labels between train and test sets are removed
- Embargo applied to prevent leakage

- **Metric:** Accuracy  
- **Result:** ~74% mean CV accuracy over 5 folds

### Diagnostics

- Confusion Matrix
- Feature Importance:
  - **MDI** (Mean Decrease Impurity)
  - **MDA** (Permutation-based importance with Purged CV)
  - **SFI** (Single Feature Importance)

---

## Backtesting

### Combinatorial Purged Cross-Validation (CPCV)

- Data split into groups
- Multiple train/test paths constructed via combinations
- Purging and embargo applied at every split

### Execution Logic

Implemented in `AFMLBacktester`:

- Event-driven trade entry
- Model prediction determines:
  - Direction
  - Position size (confidence-based)
- Volatility-based barriers constructed using a rolling estimator
- Trade lifecycle:
  - Entry
  - Barrier monitoring
  - Exit on:
    - Profit
    - Loss
    - Time expiry

Multiple simultaneous trades are supported.

---

## Results (v1)

Results are reported across multiple **Combinatorial Purged Cross-Validation (CPCV)** test regimes.
Each regime represents a distinct out-of-sample path with full data engineering, training, and execution.

### Aggregate Observations

- **Initial Capital:** 100,000
- **Final Equity Range:** ~97,500 to ~113,000
- **Per-Regime Return Range:** approximately **−2.5% to +13.1%**
- **Trades per Regime:** ~3,600 to ~4,400
- **Sharpe-like Metric:** typically between **−0.03 and 0.02**

### Best / Worst Regimes

- **Best Return:** ~+13.1%
- **Worst Return:** ~−2.5%
- **Best Sharpe-like Value:** ~0.024
- **Worst Sharpe-like Value:** ~−0.033

### Interpretation

- Performance varies significantly across regimes, indicating **strong regime dependence**
- The signal exhibits **low risk-adjusted returns** after costs
- Several regimes are marginally profitable, while others are flat or losing
- No regime-level cherry-picking is performed

These results suggest that while the signal contains **localized predictive structure**, it is **not robust enough for deployment** without further filtering, regime selection, or meta-modeling.

---

## Pipeline Architecture

The project is fully modularized into three pipelines:

1. **Data Pipeline**
2. **Model Pipeline**
3. **Backtesting Pipeline**

Each pipeline:
- Runs independently
- Saves outputs and configuration under a unique run ID
- Can be chained by passing the run ID forward

This enables reproducibility and regime-based experimentation.

---

## Reproducibility & Run Artifacts

All experimental outputs produced by this pipeline are persisted to the `runs/` directory.

Each execution of the pipeline is associated with a **unique `run_id`**, which serves as the single source of truth for that experiment.

For a given `run_id`, the corresponding directory contains:

- **Model outputs**
  - Trained model artifacts
  - Cross-validation scores
  - Confusion matrices

- **Feature diagnostics**
  - Feature importance results:
    - MDI
    - MDA
    - SFI
  - Intermediate feature matrices used for training

- **Backtesting results**
  - Per-trade logs
  - Equity curves
  - Realized and unrealized PnL
  - Regime-level performance summaries

- **Configuration files**
  - Data parameters
  - Feature engineering settings
  - Model hyperparameters
  - Backtesting and execution configs

This design enables:
- Full experiment reproducibility
- Regime-by-regime inspection
- Post-hoc analysis without rerunning the pipeline

To analyze or extend a past experiment, downstream pipelines can be executed by simply passing the corresponding `run_id`.

---

## Known Limitations (v1)

- Borrowing costs for shorts not modeled
- Spread estimation is static per regime
- Execution latency simplified
- Trade filtering intentionally permissive

All limitations are explicit and intentional.

---

## Roadmap

Planned extensions:

- Trade filtering based on expected net returns
- Regime and volatility gating
- Holding-time-aware exits
- Improved execution latency modeling
- Detailed PnL attribution and diagnostics
- Meta-model to evaluate signal quality
- Deployment via Binance trading API

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- Easley et al. — Market microstructure and order flow literature

---

## Disclaimer

This project is for **research and educational purposes only**.

It is **not financial advice** and must not be used for live trading.