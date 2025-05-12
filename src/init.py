"""
Core package for Regime Switching Models Project.
This package contains modules for data loading, feature engineering, 
model training, signal generation, backtesting, and visualization.
"""

# Import core modules for easier access
from .data_loader import DataLoader
from .feature_engineering import create_features
from .hmm_model import HMMModel
from .signal_generation import generate_signals
from .backtesting import Backtest
from .utils import configure_logging, sharpe_ratio, load_config, current_timestamp
from .plotter import plot_equity_curve, plot_regime_states, plot_signal_performance

# Define the public API for the package
__all__ = [
    "DataLoader",
    "create_features",
    "HMMModel",
    "generate_signals",
    "Backtest",
    "configure_logging",
    "sharpe_ratio",
    "load_config",
    "current_timestamp",
    "plot_equity_curve",
    "plot_regime_states",
    "plot_signal_performance",
]
