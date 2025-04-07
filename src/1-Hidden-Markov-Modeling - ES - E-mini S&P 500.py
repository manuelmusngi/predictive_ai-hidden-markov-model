# Hidden Markov Model and Market Regimes - ES E-mini S&P 500

# Library Dependencies
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import logging
import itertools
import matplotlib.pyplot as plt

from pyhhmm.gaussian import GaussianHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Supress warning in hmmlearn
warnings.filterwarnings("ignore")

# Change plot style to ggplot for better visualization
plt.style.use('ggplot')

# Futures Class
class FuturesProduct():
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.get_data()
        self.log_returns()
    
    def get_data(self):
        futures = yf.download(self.symbol, self.start_date, self.end_date)
        futures.rename(columns = {"Adj Close" : "price"}, inplace = True)
        self.data = futures
        return futures
    
    def log_returns(self):
        self.data["log_returns"] = np.log(self.data.price / self.data.price.shift(1))
        
    def plot_prices(self):
        self.data["Adj Close"].plot(figsize = (12,8), color = "blue")
        plt.title("Price Chart: {}".format(self.symbol), fontsize = 15)
        
    def plot_returns(self, kind = "ts"):
        if kind == "ts":
            self.data.log_returns.plot(figsize = (12,8))
            plt.title("Returns: {}").format(self.symbol, fontsize = 15)
        elif kind == "hist":
            self.data.log_returns.hist(figsize = (12,8), bins = int(np.sqrt(len(self.data))))
            plt.title("Return frequency: {}".format(self.symbol), fontsize = 15)

# Instrument Observation
es = FuturesProduct("ES=F", "2017-01-01", "2022-12-31")

es.get_data()
es.log_returns()
es.data.price.plot()
es.data.log_returns.plot()
es.data.log_returns.hist(bins = 100)

# Hidden Markov Model Statistical Learning
start_date = "2017-01-01"
end_date = "2022-12-31"
symbol = "ES=F"
futures = yf.download(symbol, start_date, end_date)
df = futures.copy()

# Feature Engineering
# Create features for evaluation
df["Returns"] = (df["Adj Close"] / df["Adj Close"].shift(1)) - 1
df["Volatility"] = (df["High"] / df["Low"]) - 1
df.dropna(inplace = True)

# Features for training
X_train = df[["Returns", "Volatility"]]
X_train.head()

# Hidden Markov Model Training
# Model Training
model = GaussianHMM(n_states=4, covariance_type='full', n_emissions=2)
model.train([np.array(X_train.values)])

# Regime State preliminary observations
hidden_states = model.predict([X_train.values])[0]
print(hidden_states[:40])
len(hidden_states)

# Regime State feature Means and Covariance 
model.means
model.covars

# Hidden Markov Model Result
# Regime State structure to be plotted
i = 0
regime_state_0 = []
regime_state_1 = []
regime_state_2 = []
 ▋
