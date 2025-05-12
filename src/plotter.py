import matplotlib.pyplot as plt
import pandas as pd

def plot_equity_curve(df, benchmark_col="benchmark_cumulative_expo", strategy_col="strategy_cumulative_expo"):
    """
    Plots the equity curve for both benchmark and strategy.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the equity curve data.
    - benchmark_col (str): Column name for the benchmark cumulative returns.
    - strategy_col (str): Column name for the strategy cumulative returns.

    Returns:
    - None: Displays the plot.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(df[benchmark_col], label="Benchmark", color="blue", linewidth=2)
    plt.plot(df[strategy_col], label="Strategy", color="green", linewidth=2)
    plt.title("Equity Curve", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Returns", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def plot_regime_states(df, regime_col="HMM", price_col="Adj Close"):
    """
    Plots the market regime states overlaid on the price data.

    Parameters:
    - df (pd.DataFrame): DataFrame containing price and regime state data.
    - regime_col (str): Column name for the hidden Markov model's regime states.
    - price_col (str): Column name for the price data.

    Returns:
    - None: Displays the plot.
    """
    plt.figure(figsize=(14, 8))
    for regime in df[regime_col].unique():
        regime_data = df[df[regime_col] == regime]
        plt.plot(regime_data.index, regime_data[price_col], label=f"Regime {regime}")

    plt.title("Market Regime States", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def plot_signal_performance(df, signal_col="Main_Signal", price_col="Adj Close"):
    """
    Plots the trading signals on the price chart.

    Parameters:
    - df (pd.DataFrame): DataFrame containing price and signal data.
    - signal_col (str): Column name for the trading signals.
    - price_col (str): Column name for the price data.

    Returns:
    - None: Displays the plot.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(df[price_col], label="Price", color="blue", linewidth=2)
    buy_signals = df[df[signal_col] == 1]
    plt.scatter(buy_signals.index, buy_signals[price_col], label="Buy Signal", color="green", marker="^", alpha=1.0)
    plt.title("Trading Signals", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
