import os
from dataclasses import dataclass, field
from typing import List, Any


@dataclass
class DataConfig:
    """
    Configuration class for the data pipeline.
    Contains parameters for fetching, bar generation,
    feature engineering, labeling, weighting, bootstrapping,
    and fractional differencing.
    """
    # general parameters for data fetching 
    asset: str = "BTCUSDT"    # the sign and name of crypto
    type: str = "trade"       # e.g., 'trades' or 'order_book'
    interval_ms: int = 1000   # the time interval between trades
    nRows: int = 0            # number of rows to fetch

    bar_type: str = "volume"    # e.g., 'tick', 'time', 'volume'
    threshold: int = 0.05                                   

    # features enginering
    window: int = 30               # rolling window size for features

    # labeling the dataset using triple barrier method
    pt_sl: List[float] = field(default_factory=lambda: [1, 1])  # [stop loss, profit cap]
    min_ret: float = 0                                        # minimum return
    event_specific: bool = True                                
    h: int = 30                                                 # holding period (number of bars ahead)

    # to generate sample weights 
    clfLastW: float = 0.8    # how to much to decay accorrding to time (1 -> no decay | 0 -> complete decay)

    sLength: int = 5000      # number of samples to bootstrap
    
    # fractional differentiation
    fracdiff_cols: List[str] = field(default_factory=lambda: ["close"]) # columns to frac diff
    d: float = 0.7                       # how much to (1-> complete | 0 -> no diff)
    thres: float = 0.04                   

    data_dir: str = "data"
    log_file: str = "pipeline.log"

    # validate
    def validate(self) -> None:
        """Validate critical parameters and directory structure."""
        assert self.type in ["trades", "depth"], f"Invalid data type: {self.type}"
        assert self.bar_type in ["tick", "time", "volume"], f"Invalid bar type: {self.bar_type}"
        assert 0 < self.window, "Window must be positive"
        assert 0 < self.h, "Holding period must be positive"
        assert 0 < self.threshold, "Threshold must be positive"
        assert 0 < self.d < 1, "Fractional differencing 'd' must be between 0 and 1"

        # Ensure data directory exists
        os.makedirs(os.path.join(self.data_dir, self.asset), exist_ok=True)

    def summary(self) -> None:
        """Print a formatted summary of the configuration."""
        print("\n=== PIPELINE CONFIGURATION ===")
        for k, v in self.__dict__.items():
            print(f"{k:20}: {v}")
        print("===============================\n")


@dataclass
class ModelConfig:
    """Configuration for model training, evaluation, and feature importance."""

    model_params = {'n_estimators':100, 'criterion':'gini', 'max_depth':None, 
                    'min_samples_split':2, 'min_samples_leaf':1}
    
    model: str = "tree"                  # 'tree', 'xgb', 'lgbm', 'catboost', etc.
    scoring: str = "neg_log_loss"            # metric for evaluation
    cv: int = 5                          # number of Purged K-Folds
    pctEmbargo: float = 0.01             # percent embargo between folds
    labels: List[Any] = field(default_factory=lambda: [0, 1, -1])  # depends on labeling

    # Directories
    model_dir: str = "models"
    result_dir: str = "results"
    log_dir: str = "logs"

    def validate(self) -> None:
        assert self.cv >= 2, "cv must be at least 2"
        assert 0 <= self.pctEmbargo < 1, "pctEmbargo must be in [0, 1)"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def summary(self) -> None:
        print("\n=== MODEL PIPELINE CONFIG ===")
        for k, v in self.__dict__.items():
            print(f"{k:20}: {v}")
        print("=============================\n")


@dataclass
class BTConfig:
    cpcv_splits = 6
    cpcv_test_splits = 2
    cpcv_embargo = 0.01
    purge_window = 30

    lookback_vol = 30
    K_pt_base =  2.5
    K_sl_base =  1.5
    alpha = 0.5
    horizon_bars = 30
    initial_equity = 100000.0
    risk_per_trade = 0.01
    base_unit = 1.0
    max_open_trades = 100
    event_threshold = 0.05
    max_size:int=30
    min_size:int=0.01
    k:int=1.25
    lamda:int=0.2
    f:int=0.0005
    borrow_rate:int=0.08


# Example usage
if __name__ == "__main__":
    configs = DataConfig()
    configs.validate()
    configs.summary()