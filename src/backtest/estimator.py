import math
import numpy as np
import pandas as pd

# --------------------------- Volatility Estimator ---------------------------
class VolatilityEstimator:
    """Simple volatility utilities: rolling std of log returns and ATR.

    Use whichever fits your frequency/data. This class caches rolling calculations for speed.
    """

    def __init__(self, prices: pd.Series, lookback: int = 20):
        self.prices = prices
        self.lookback = lookback
        self._logret = np.log(prices).diff()
        self._sigma = self._logret.rolling(window=lookback, min_periods=1).std()

    def sigma_at(self, idx: int) -> float:
        # returns sigma (std of log returns) at integer index
        val = float(self._sigma.iloc[idx])
        if math.isnan(val) or val <= 0:
            # fallback tiny volatility to avoid degenerate barriers
            return 1e-6
        return val

    @staticmethod
    def atr_at(high: pd.Series, low: pd.Series, close: pd.Series, idx: int, lookback: int = 14) -> float:
        # optional: compute ATR at idx using pandas - simplified
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=lookback, min_periods=1).mean()
        val = float(atr.iloc[idx])
        if math.isnan(val) or val <= 0:
            return 1e-6
        return val