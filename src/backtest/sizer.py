from typing import Optional

class PositionSizer:
    def __init__(self, initial_equity: float, risk_per_trade: float = 0.01, max_size: Optional[float] = None):
        self.equity = initial_equity
        self.risk_per_trade = risk_per_trade  # fraction of equity
        self.max_size = max_size

    def size_from_risk(self, entry_price: float, sl_price: float) -> float:
        # compute absolute size in asset units so that max loss = risk_per_trade * equity
        d = abs(entry_price - sl_price)
        if d <= 0:
            return 0.0
        allowed_loss = self.risk_per_trade * self.equity
        size = allowed_loss / d
        if self.max_size is not None:
            return min(size, self.max_size)
        return size

    def update_equity(self, realized_pnl: float):
        self.equity += realized_pnl

if __name__ == "__main__":pass