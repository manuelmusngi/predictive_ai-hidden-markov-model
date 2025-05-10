import numpy as np

class Backtest:
    def __init__(self, data, risk_free_rate, annual_trading_days):
        self.data = data
        self.rf = risk_free_rate
        self.N = annual_trading_days

    def calculate_returns(self):
        self.data["benchmark_log_return"] = np.log(self.data["Adj Close"] / self.data["Adj Close"].shift(1))
        self.data["strategy_log_return"] = np.log(self.data["Open"].shift(-1) / self.data["Open"]) * self.data["Main_Signal"]
        self.data["benchmark_cumulative"] = self.data["benchmark_log_return"].cumsum()
        self.data["strategy_cumulative"] = self.data["strategy_log_return"].cumsum()

    def sharpe_ratio(self, returns):
        mean = returns.mean() * self.N
        sigma = returns.std() * np.sqrt(self.N)
        return round((mean - self.rf) / sigma, 2)

    def evaluate(self):
        self.calculate_returns()
        benchmark_sharpe = self.sharpe_ratio(self.data["benchmark_log_return"])
        strategy_sharpe = self.sharpe_ratio(self.data["strategy_log_return"])
        return benchmark_sharpe, strategy_sharpe
