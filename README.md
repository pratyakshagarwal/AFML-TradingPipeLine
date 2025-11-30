# AFML-TradingPipeline (In Progress)

A modular, research-grade trading pipeline inspired by `Advances in Financial Machine Learning`.

The system is structured into four independent yet tightly connected stages:

1. Data Handling
Includes the full data-engineering workflow for training ML-driven trading models:

- Crypto market data acquisition (trades + order book)
- Feature engineering and order-book transformations
- Triple-barrier labeling
- Sample weighting using average uniqueness and time-decay
- Sequential bootstrapping (Numba-accelerated) based on average uniqueness

2. Training / Evaluation / Hyperparameter Tuning
Implements a reproducible experimentation framework with:

- Custom datasets and dataloaders
- Model training loops
- Evaluation metrics (PnL, Sharpe, drawdown, hit-rate, etc.)
- Config-driven hyperparameter tuning

3. Backtesting Engine (IN Progress)
4. Infrencing/Live Simulation (IN Progress)