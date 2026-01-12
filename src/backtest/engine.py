from __future__ import annotations

import math
import uuid
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

from src.backtest.estimator import VolatilityEstimator
from src.backtest.sizer import PositionSizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("afml_backtester")

# =========================== TRADE ===========================

@dataclass
class Trade:
    trade_id: str
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    side: int
    conf: float
    size: float
    pt: float
    sl: float
    expiry_idx: int
    vol_at_entry: float
    fee_entry: float

    def to_dict(self):
        return asdict(self)


# =========================== UTILS ===========================

def new_trade_id() -> str:
    return uuid.uuid4().hex


def compute_pnl(
    trade: Trade,
    exit_price: float,
    fee_exit: float,
) -> float:
    gross = trade.size * trade.side * (exit_price - trade.entry_price)
    return gross - trade.fee_entry - fee_exit


# =========================== BACKTESTER ===========================

class AFMLBacktester:

    def __init__(self, path: pd.DataFrame, config, model):
        self.path = path.reset_index(drop=True)
        self.config = config
        self.model = model

        self.trades_active: Dict[str, Trade] = {}
        self.trades_closed: Dict[str, Dict[str, Any]] = {}

        self.diff = self.path["close"].diff().fillna(0.0)
        self.log_ret = np.diff(np.log(self.path["close"].values))

        self.s_pos, self.s_neg = 0.0, 0.0

        self.vol_est = VolatilityEstimator(
            self.path["close"],
            lookback=config.lookback_vol
        )

        self.sizer = PositionSizer(
            initial_equity=config.initial_equity,
            risk_per_trade=config.risk_per_trade,
            max_size=config.max_size
        )

        self.rspread = self._estimate_roll_spread()

    # ================= CORE LOOP =================

    def run(self):
        for idx, row in self.path.iterrows():
            price = float(row["close"])
            ts = row["timestamp"]

            if self.is_event(idx):
                self.open_trade(idx, ts, price)

            closed = self.update_trades(idx, ts, price)
            for info in closed.values():
                self.sizer.update_equity(info["realized_pnl"])

        return self.summary()

    # ================= EVENT =================

    def is_event(self, idx: int) -> bool:
        self.s_pos = max(0.0, self.s_pos + self.diff.iloc[idx])
        self.s_neg = min(0.0, self.s_neg + self.diff.iloc[idx])

        if abs(self.s_pos) > self.config.event_threshold or abs(self.s_neg) > self.config.event_threshold:
            self.s_pos, self.s_neg = 0.0, 0.0
            return True
        return False

    # ================= MODEL =================

    def model_predict(self, row: pd.Series) -> Tuple[int, float]:
        X = row.drop("timestamp").to_frame().T
        proba = self.model.predict_proba(X)[0]
        cls = self.model.classes_[np.argmax(proba)]
        return int(cls), float(np.max(proba))

    # ================= TRADE LIFECYCLE =================

    def open_trade(self, idx: int, ts: pd.Timestamp, price: float):
        signal, conf = self.model_predict(self.path.iloc[idx])
        if signal == 0 or len(self.trades_active) >= self.config.max_open_trades:
            return

        vol = self.vol_est.sigma_at(idx)
        spread = max(self.rspread, vol * self.config.k)
        slippage = vol * self.config.lamda

        pt, sl, expiry_idx = self.make_barriers(idx, price, signal, conf, vol)
        size = self.position_size(price, sl, conf)
        if size <= 0:
            return

        fee_entry, exec_entry = self._execution_fee(price, size, spread, slippage)

        trade = Trade(
            trade_id=new_trade_id(),
            entry_idx=idx,
            entry_time=ts,
            entry_price=exec_entry,
            side=signal,
            conf=conf,
            size=size,
            pt=pt,
            sl=sl,
            expiry_idx=expiry_idx,
            vol_at_entry=vol,
            fee_entry=fee_entry
        )

        self.trades_active[trade.trade_id] = trade

    def update_trades(self, idx: int, ts: pd.Timestamp, price: float):
        closed = {}

        for tid, trade in list(self.trades_active.items()):
            hit_pt = price >= trade.pt if trade.side == 1 else price <= trade.pt
            hit_sl = price <= trade.sl if trade.side == 1 else price >= trade.sl
            hit_expiry = idx >= trade.expiry_idx

            if hit_pt or hit_sl or hit_expiry:
                vol = self.vol_est.sigma_at(idx)
                spread = max(self.rspread, vol * self.config.k)
                slippage = vol * self.config.lamda

                fee_exit, exec_price  = self._execution_fee(price, trade.size, spread, slippage)
                # borrow_cost = self._borrow_cost(trade, idx)

                pnl = compute_pnl(trade, exec_price, fee_exit)

                closed[tid] = {
                    "trade": trade.to_dict(),
                    "exit_time": ts,
                    "exit_price": exec_price,
                    "realized_pnl": pnl,
                    "reason": "pt" if hit_pt else ("sl" if hit_sl else "expiry"),
                }

                del self.trades_active[tid]

        self.trades_closed.update(closed)
        return closed

    # ================= LOGIC =================

    def make_barriers(self, idx, price, side, conf, vol):
        k_pt = self.config.K_pt_base * (1 + self.config.alpha * (conf - 0.5))
        k_sl = self.config.K_sl_base * (1 + self.config.alpha * (conf - 0.5))

        pt = price * math.exp(k_pt * vol * side)
        sl = price * math.exp(-k_sl * vol * side)

        hb = int(self.config.horizon_bars)
        expiry_idx = min(idx + hb, len(self.path) - 1)

        return pt, sl, expiry_idx

    def position_size(self, entry_price, sl_price, conf):
        size_risk = self.sizer.size_from_risk(entry_price, sl_price)
        size_conf = self.config.base_unit * conf
        return max(self.config.min_size, min(size_risk, size_conf))

    # ================= COSTS =================

    def _estimate_roll_spread(self):
        if len(self.log_ret) < 3:
            return 0.0

        r_t = self.log_ret[1:]
        r_tm1 = self.log_ret[:-1]
        cov = np.cov(r_t, r_tm1, bias=True)[0, 1]

        return 2 * np.sqrt(-cov) if cov < 0 else 0.0

    def _execution_fee(self, price, Q, spread, slippage):
        exec_price = price * (spread / 2 + slippage)
        return exec_price * Q * self.config.f, exec_price

    def _borrow_cost(self, trade: Trade, exit_idx: int):
        if trade.side != -1:
            return 0.0

        bars_held = exit_idx - trade.entry_idx
        rate_per_bar = self.config.borrow_rate / self.config.bars_per_year

        notional = trade.entry_price * trade.size
        return notional * rate_per_bar * bars_held

    def summary(self):
        if not self.trades_closed:
            return {
                'n_closed': 0,
                'n_active': len(self.trades_active),
                'equity': self.sizer.equity
            }

        df = pd.DataFrame([
            {
                'trade_id': tid,
                'entry_time': info['trade']['entry_time'],
                'exit_time': info['exit_time'],
                'entry_price': info['trade']['entry_price'],
                'exit_price': info['exit_price'],
                'side': info['trade']['side'],
                'size': info['trade']['size'],
                'realized_pnl': info['realized_pnl'],
                'reason': info['reason']
            }
            for tid, info in self.trades_closed.items()
        ])

        returns = df['realized_pnl'] / (df['entry_price'] * df['size']).replace(0, np.nan)
        returns = pd.Series(returns)

        # hard sanitize
        returns = returns.apply(
            lambda x: float(x[0]) if isinstance(x, (np.ndarray, list)) else float(x)
        )
        return {
            'n_closed': len(df),
            'n_active': len(self.trades_active),
            'equity': self.sizer.equity,
            'total_realized_pnl': df['realized_pnl'].sum(),
            'avg_pnl': df['realized_pnl'].mean(),
            'sharpe_like' : (
                returns.mean() / returns.std()
                if len(returns) > 1 and returns.std() != 0
                else None
            ),
        }, df


# =========================== EXAMPLE ===========================

if __name__ == "__main__":
    n = 1000
    rng = pd.date_range("2020-01-01", periods=n, freq="T")
    price = 100 + np.cumsum(np.random.randn(n) * 0.1)

    df = pd.DataFrame({
        "timestamp": rng,
        "price": price
    })

    class MyBacktester(AFMLBacktester):
        def model_predict(self, row: pd.Series) -> Tuple[int, float]:
            idx = int(row.name)
            if idx < 5:
                return 0, 0.0

            mom = price[idx] - price[max(0, idx - 5)]
            if abs(mom) < 0.01:
                return 0, 0.0

            signal = 1 if mom > 0 else -1
            conf = min(1.0, abs(mom) / 0.2)
            return signal, conf
        
    from config import BTConfig
    config = BTConfig(
    )

    bt = MyBacktester(df, config)
    res = bt.run()
    print(res)
