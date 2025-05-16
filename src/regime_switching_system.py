import os
import logging
import yaml
import pandas as pd
import numpy as np
import yfinance as yf
from pyhhmm.gaussian import GaussianHMM
import matplotlib.pyplot as plt

# --- Utility Functions (could be moved to utils.py) ---

def configure_logging(log_file="logs/app.log"):
    if not os.path.exists("logs/"):
        os.makedirs("logs/")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging is configured.")

def sharpe_ratio(returns_series, risk_free_rate=0.01, annual_trading_days=255):
    mean_return = returns_series.mean() * annual_trading_days
    std_dev = returns_series.std() * np.sqrt(annual_trading_days)
    sharpe = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else 0
    return round(sharpe, 2)

def load_config(config_path="config/config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# --- Main System Class ---

class RegimeSwitchingSystem:
    def __init__(self, config_path="config/config.yaml"):
        configure_logging()
        self.config = load_config(config_path)
        self.symbol = self.config["data"]["symbol"]
        self.start_date = self.config["data"]["start_date"]
        self.end_date = self.config["data"]["end_date"]
        self.raw_data_path = self.config["data"]["raw_data_path"]
        self.processed_data_path = self.config["data"]["processed_data_path"]
        self.model_params = self.config["model"]
        self.backtest_params = self.config["backtesting"]
        self.data = None
        self.features = None
        self.model = None
        self.signals_df = None

    # 1. Data Loading
    def fetch_data(self):
        raw_file = os.path.join(self.raw_data_path, f"{self.symbol}.csv")
        if os.path.exists(raw_file):
            logging.info(f"Loading pre-saved data from {raw_file}")
            self.data = pd.read_csv(raw_file, index_col="Date", parse_dates=True)
        else:
            logging.info(f"Downloading data for {self.symbol} from Yahoo Finance")
            self.data = yf.download(self.symbol, self.start_date, self.end_date)
            if not os.path.exists(self.raw_data_path):
                os.makedirs(self.raw_data_path)
            self.data.to_csv(raw_file)
        return self.data

    # 2. Feature Engineering
    def create_features(self):
        df = self.data.copy()
        # Use Adj Close for returns and moving averages; fallback if not present
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        df["Returns"] = (df[price_col] / df[price_col].shift(1)) - 1
        df["Volatility"] = (df["High"] / df["Low"]) - 1
        df["MA_9"] = df[price_col].rolling(window=9).mean()
        df["MA_21"] = df[price_col].rolling(window=21).mean()
        df.dropna(inplace=True)
        self.features = df
        # Save processed data
        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)
        processed_file = os.path.join(self.processed_data_path, f"{self.symbol}_features.csv")
        df.to_csv(processed_file)
        return df

    # 3. Model Training
    def train_hmm(self, train_size=500):
        X_train = self.features[["Returns", "Volatility"]].iloc[:train_size]
        self.model = GaussianHMM(
            n_states=self.model_params["n_states"],
            covariance_type=self.model_params["covariance_type"],
            n_emissions=self.model_params["n_emissions"]
        )
        self.model.train([np.array(X_train.values)])
        logging.info("HMM model trained.")
        return self.model

    # 4. Signal Generation
    def generate_signals(self, train_size=500):
        X_test = self.features[["Returns", "Volatility"]].iloc[train_size:]
        save_df = self.features.iloc[train_size:].copy()
        hmm_results = self.model.predict([np.array(X_test.values)])[0]

        save_df["HMM"] = hmm_results
        save_df["MA_Signal"] = (save_df["MA_9"] > save_df["MA_21"]).astype(int)
        # Favorable states from config
        favorable_states = self.backtest_params["favorable_states"]
        save_df["HMM_Signal"] = save_df["HMM"].apply(lambda x: 1 if x in favorable_states else 0)
        save_df["Main_Signal"] = ((save_df["MA_Signal"] == 1) & (save_df["HMM_Signal"] == 1)).astype(int).shift(1)
        self.signals_df = save_df
        logging.info("Signals generated.")
        return save_df

    # 5. Backtesting
    def backtest(self):
        df = self.signals_df.copy()
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        df["benchmark_log_return"] = np.log(df[price_col] / df[price_col].shift(1))
        df["strategy_log_return"] = np.log(df["Open"].shift(-1) / df["Open"]) * df["Main_Signal"]
        df["benchmark_cumulative"] = df["benchmark_log_return"].cumsum()
        df["strategy_cumulative"] = df["strategy_log_return"].cumsum()
        df["benchmark_cumulative_expo"] = np.exp(df["benchmark_cumulative"]) - 1
        df["strategy_cumulative_expo"] = np.exp(df["strategy_cumulative"]) - 1
        df.dropna(inplace=True)

        benchmark_return = round(df["benchmark_cumulative_expo"].values[-1] * 100, 1)
        benchmark_sharpe = sharpe_ratio(df["benchmark_log_return"], risk_free_rate=self.backtest_params["risk_free_rate"])
        strategy_return = round(df["strategy_cumulative_expo"].values[-1] * 100, 1)
        strategy_sharpe = sharpe_ratio(df["strategy_log_return"], risk_free_rate=self.backtest_params["risk_free_rate"])

        results = {
            "df": df,
            "benchmark_return": benchmark_return,
            "benchmark_sharpe": benchmark_sharpe,
            "strategy_return": strategy_return,
            "strategy_sharpe": strategy_sharpe,
        }
        logging.info("Backtesting complete.")
        return results

    # 6. Plotting
    def plot_equity_curve(self, df):
        plt.figure(figsize=(12, 8))
        plt.plot(df["benchmark_cumulative_expo"], label="Benchmark", color="blue", linewidth=2)
        plt.plot(df["strategy_cumulative_expo"], label="Strategy", color="green", linewidth=2)
        plt.title("Equity Curve", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Cumulative Returns", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_regime_states(self, df, regime_col="HMM", price_col="Adj Close"):
        plt.figure(figsize=(14, 8))
        for regime in df[regime_col].unique():
            regime_data = df[df[regime_col] == regime]
            plt.plot(regime_data.index, regime_data[price_col], label=f"Regime {regime}")
        plt.title("Market Regime States", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_signal_performance(self, df, signal_col="Main_Signal", price_col="Adj Close"):
        plt.figure(figsize=(12, 8))
        plt.plot(df[price_col], label="Price", color="blue", linewidth=2)
        buy_signals = df[df[signal_col] == 1]
        plt.scatter(buy_signals.index, buy_signals[price_col], label="Buy Signal", color="green", marker="^", alpha=1.0)
        plt.title("Trading Signals", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    # 7. End-to-End Pipeline
    def run_pipeline(self):
        self.fetch_data()
        self.create_features()
        self.train_hmm()
        self.generate_signals()
        results = self.backtest()
        print(f"Benchmark Returns: {results['benchmark_return']}%")
        print(f"Benchmark Sharpe: {results['benchmark_sharpe']}")
        print(f"Strategy Returns: {results['strategy_return']}%")
        print(f"Strategy Sharpe: {results['strategy_sharpe']}")
        # Plot
        self.plot_equity_curve(results["df"])
        self.plot_regime_states(results["df"])
        self.plot_signal_performance(results["df"])
        return results

# Example usage (in main.py):
# from src.regime_switching_system import RegimeSwitchingSystem
# system = RegimeSwitchingSystem(config_path="config/config.yaml")
# system.run_pipeline()
