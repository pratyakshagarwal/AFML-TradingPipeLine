import numpy as np
import pandas as pd
from typing import Literal

class BarGenerator:
    """
    Generates bars (volume, tick, or dollar) from trade data.
    Designed for extensibility and performance.
    """

    def __init__(self, threshold: float, bar_type: Literal["volume", "tick", "dollar"] = "volume"):
        self.threshold = threshold
        self.bar_type = bar_type

    def _get_metric(self, price: float, size: float) -> float:
        if self.bar_type == "volume":
            return size
        elif self.bar_type == "tick":
            return 1.0
        elif self.bar_type == "dollar":
            return price * size
        else:
            raise ValueError(f"Invalid bar_type: {self.bar_type}")

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"price", "quantity", "trade_time"}
        missing = required_cols - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        bars = []
        open_, high, low, close, volume = np.nan, -np.inf, np.inf, np.nan, 0.0

        for price, qty, ts in zip(data["price"], data["quantity"], data["trade_time"]):
            if np.isnan(open_):
                open_ = price
            high = max(high, price)
            low = min(low, price)
            volume += self._get_metric(price, qty)

            if volume >= self.threshold:
                close = price
                bars.append([ts, open_, high, low, close, volume])
                open_, high, low, close, volume = np.nan, -np.inf, np.inf, np.nan, 0.0

        if not np.isnan(open_):
            ts = data["trade_time"].iloc[-1]
            bars.append([ts, open_, high, low, close, volume])

        bars_df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], unit="ms")
        # bars_df.set_index("timestamp", inplace=True)
        return bars_df


def getWeights_FFD(d, thres):
    # thres>0 drops insignificant weights
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] /k* (d - k + 1) 
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def fracDiff_FFD(series,d,thres=1e-5):
    '''
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    #1) Compute weights for the longest series
    w=getWeights_FFD(d, thres)
    width=len(w)-1
    #2) Apply weights to values
    df={}
    for name in series.columns:
        seriesF,df_=series[[name]].fillna(method='ffill').dropna(),pd.Series()
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width],seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):continue # exclude NAs
            df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df


if __name__ == "__main__":
    import os
    for name in os.listdir('data/btcusdt'):
        if name.endswith('.jsonl'):
            path = os.path.join('data/btcusdt', name)
            df = pd.read_json(path, lines=True)
            print(df.head())
            df = df.drop_duplicates(subset=['trade_id'])
            print(len(df))

            bg = BarGenerator(threshold=0.05, bar_type="volume")
            bar_df = bg.generate(df)
            print(bar_df.head())

            date = name.split('_')[0]
            bar_df.to_csv(f"data/btcusdt/{date}_tradebars.csv", index=False)
