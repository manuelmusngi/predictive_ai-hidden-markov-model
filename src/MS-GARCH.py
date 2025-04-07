import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
from hmmlearn.hmm import GaussianHMM

# Step 1: Fetch NQ Data
def fetch_nq_data():
    nq_data = yf.download('^NDX', start='2000-01-01', end='2025-01-01')
    return nq_data

# Step 2: Preprocess Data
def preprocess_data(data):
    data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)
    return data

# Step 3: Fit GARCH Model
def fit_garch_model(returns):
    garch_model = arch_model(returns, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp='off')
    return garch_fit

# Step 4: Fit MS-GARCH Model
def fit_ms_garch_model(data):
    returns = data['Returns'].values
    garch_fit = fit_garch_model(returns)
    
    # Extract residuals and standardize them
    residuals = garch_fit.resid / garch_fit.conditional_volatility
    residuals = residuals.values.reshape(-1, 1)

    # Fit HMM to standardized residuals
    hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000)
    hmm.fit(residuals)
    
    return hmm, garch_fit

# Step 5: Main function to run the model
def run_ms_garch():
    nq_data = fetch_nq_data()
    nq_data = preprocess_data(nq_data)
    hmm, garch_fit = fit_ms_garch_model(nq_data)
    
    print("GARCH Model Summary:")
    print(garch_fit.summary())
    
    return hmm, garch_fit

# Execute the function
if __name__ == "__main__":
    run_ms_garch()
