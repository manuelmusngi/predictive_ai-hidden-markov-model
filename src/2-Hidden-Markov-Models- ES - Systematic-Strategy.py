# Library Dependencies
import pandas as pd
import numpy as np

from pyhhmm.gaussian import GaussianHMM
import yfinance as yf

import matplotlib.pyplot as plt

# Data Import
start_date = "2017-01-1"
end_date = "2024-06-1"
symbol = "ES=F"
data = yf.download(symbol, start_date, end_date)

# Returns and Volatility Features
df = data.copy()
df["Returns"] = (df["Adj Close"] / df["Adj Close"].shift(1)) - 1
df["Volatility"] = (df["High"] / df["Low"]) - 1
df.dropna(inplace=True)
print("Length: ", len(df))
df.head()

# Moving Average Features
df["MA_9"] = df["Adj Close"].rolling(window=9).mean()
df["MA_21"] = df["Adj Close"].rolling(window=21).mean()

# Structure Data for Training and Testing
X_train = df[["Returns", "Volatility"]].iloc[:500] 
X_test = df[["Returns", "Volatility"]].iloc[500:] 
save_df = df.iloc[500:]

print("Train Length: ", len(X_train))
print("Test Length: ", len(X_test))
print("X_train From: ", X_train.head(1).index.item())
print("X_train To: ", X_train.tail(1).index.item())
print("X_test From: ", X_test.head(1).index.item())
print("X_test To: ", X_test.tail(1).index.item())

# HMM Training
hmm_model = GaussianHMM(n_states=4, covariance_type='full', n_emissions=2)
hmm_model.train([np.array(X_train.values)])
hmm_model.predict([X_train.values])[0][:10]

# Test Data Prediction 
df_main = save_df.copy()
df_main.drop(columns=["High", "Low"], inplace=True)

hmm_results = hmm_model.predict([X_test.values])[0]
df_main["HMM"] = hmm_results
df_main.head()

# Moving Average Signals
df_main.loc[df_main["MA_9"] > df_main["MA_21"], "MA_Signal"] = 1
df_main.loc[df_main["MA_9"] <= df_main["MA_21"], "MA_Signal"] = 0

# Hidden Markov Model Signals
favorable_states = [0, 1]
hmm_values = df_main["HMM"].values
hmm_values = [1 if x in favorable_states else 0 for x in hmm_values]
df_main["HMM_Signal"] = hmm_values

# Signal Combination
df_main["Main_Signal"] = 0
df_main.loc[(df_main["MA_Signal"] == 1) & (df_main["HMM_Signal"] == 1), "Main_Signal"] = 1
df_main["Main_Signal"] = df_main["Main_Signal"].shift(1)

# Benchmark Returns
df_main["benchmark_log_return"] = np.log(df_main["Adj Close"] / df_main["Adj Close"].shift(1))
df_main["benchmark_cumulative"] = df_main["benchmark_log_return"].cumsum()
df_main["benchmark_cumulative_expo"] = np.exp(df_main["benchmark_cumulative"]) - 1

# Strategy Returns
df_main["strategy_log_return"] = np.log(df_main["Open"].shift(-1) / df_main["Open"]) * df_main["Main_Signal"]
df_main["strategy_cumulative"] = df_main["strategy_log_return"].cumsum()
df_main["strategy_cumulative_expo"] = np.exp(df_main["strategy_cumulative"]) - 1

# Examine Results
df_main.dropna(inplace=True)
df_main.tail()

# Sharpe Ratio Function
def sharpe_ratio(returns_series):
    N = 255
    NSQRT = np.sqrt(N)
    rf = 0.01
    mean = returns_series.mean() * N
    sigma = returns_series.std() * NSQRT
    sharpe_ratio = round((mean - rf) / sigma, 2)
    return sharpe_ratio

# Benchmark Metrics
benchmark_return = round(df_main["benchmark_cumulative_expo"].values[-1] * 100, 1)
benchmark_sharpe = sharpe_ratio(df_main["benchmark_log_return"].values)

# Strategy Metrics
strategy_return = round(df_main["strategy_cumulative_expo"].values[-1] * 100, 1)
strategy_sharpe = sharpe_ratio(df_main["strategy_log_return"].values)

# Print Metrics
print(f"Benchmark Returns: {benchmark_return}%")
print(f"Benchmark Sharpe: {benchmark_sharpe}")
print("--------------------------")
print(f"Strategy Returns: {strategy_return}%")
print(f"Strategy Sharpe: {strategy_sharpe}")

# Equity Curve 
fig = plt.figure(figsize = (18, 10))
plt.plot(df_main["benchmark_cumulative_expo"])
plt.plot(df_main["strategy_cumulative_expo"])
plt.show()
