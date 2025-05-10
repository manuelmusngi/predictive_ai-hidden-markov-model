import pandas as pd

def create_features(data):
    df = data.copy()
    df["Returns"] = (df["Adj Close"] / df["Adj Close"].shift(1)) - 1
    df["Volatility"] = (df["High"] / df["Low"]) - 1
    df["MA_9"] = df["Adj Close"].rolling(window=9).mean()
    df["MA_21"] = df["Adj Close"].rolling(window=21).mean()
    df.dropna(inplace=True)
    return df
