#### AFML Trading Pipeline (v1)

### Overview

This repository implements a research-grade algorithmic trading pipeline inspired by the methodology in Advances in Financial Machine Learning (Marcos López de Prado).

The primary goal of this project is not to present a profitable trading strategy, but to build a realistic, leakage-aware experimental framework for short-horizon alpha research under transaction costs.

Version 1 focuses on correctness, causality, and execution realism rather than performance optimization.

### Core Design Principles

- Event-based sampling (non-uniform time)
- Strict prevention of look-ahead bias
- Cost-aware execution modeling
- Walk-forward / regime-based evaluation
- Separation of signal generation and execution logic

### Pipeline Components
1. Data Processing

- Volume-bar construction from trade data
- Forward-filled features with strict causal alignment
- No future information leakage at any stage

2. Event Detection

- CUSUM-based event triggering
- Events define when the model is allowed to act
- Reduces redundant model evaluation in noisy regions

3. Labeling & Barriers

    Triple-barrier style exits:

    - Profit-taking
    - Stop-loss
    - Time-based expiry

- Barriers scale with estimated volatility

- Horizon defined in event bars, not clock time

4. Volatility Estimation

- Rolling, causal volatility estimator

    Used for:

    - barrier placement
    - execution cost scaling
    - regime analysis

5. Position Sizing

- Risk-based sizing with hard caps

    Size depends on:

    - stop-loss distance
    - model confidence

6. Execution & Costs

    Explicit modeling of:

    - exchange fees
    - bid–ask spread (Roll estimator)
    - volatility-scaled slippage

- PnL computed from observed prices, with costs applied at execution

7. Backtesting

- Trade lifecycle simulation
- Multiple simultaneous trades supported
- Walk-forward regime evaluation
- Sequential bootstrapping for robustness analysis


### What This Project Is (and Is Not)

- This project is:
    - A research framework for realistic alpha discovery
    - Suitable for microstructure and short-horizon studies
    - Designed to expose weak or fragile signals

- This project is NOT:

    - A ready-to-deploy trading system
    - A claim of consistent profitability
    - Optimized for Sharpe or capital efficiency


### Known Limitations (v1)

- Borrowing costs for short positions are not yet modeled
- Spread estimation is static per regime
- Execution latency is simplified
- Trade selection is intentionally permissive
- These limitations are documented and will be addressed incrementally.

## Roadmap

- Planned improvements for future versions:

    - Trade selectivity based on expected net returns
    - Volatility and regime gating
    - Holding-time-aware exits
    - More robust execution delay modeling
    - Detailed PnL attribution analysis

## References

- López de Prado, M. (2018). Advances in Financial Machine Learning
- Easley et al. — Microstructure and order flow literature

## Disclaimer

This project is for research and educational purposes only.
It is not financial advice and should not be used for live trading.