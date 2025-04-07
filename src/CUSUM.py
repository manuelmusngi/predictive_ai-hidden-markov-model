import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_synthetic_data(n=1000, mu=0, sigma=1):
    """Generate synthetic financial data."""
    np.random.seed(42)
    returns = np.random.normal(mu, sigma, n)
    return pd.DataFrame(returns, columns=['returns'])

def cusum(data, threshold=1, drift=0):
    """CUSUM algorithm to detect shifts in mean."""
    s_pos, s_neg = 0, 0
    pos, neg = [], []
    for x in data:
        s_pos = max(0, s_pos + x - drift)
        s_neg = max(0, s_neg - x - drift)
        pos.append(s_pos)
        neg.append(s_neg)
        if s_pos > threshold:
            s_pos = 0
        if s_neg > threshold:
            s_neg = 0
    return pos, neg

def plot_cusum(data, pos, neg, threshold):
    """Plot CUSUM results."""
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, pos, label='CUSUM Positive')
    plt.plot(data.index, neg, label='CUSUM Negative')
    plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
    plt.axhline(-threshold, color='red', linestyle='--')
    plt.title('CUSUM Analysis')
    plt.xlabel('Time')
    plt.ylabel('CUSUM')
    plt.legend()
    plt.show()

def sequential_analysis(data, window_size=30):
    """Sequential analysis to detect shifts in volatility."""
    rolling_std = data.rolling(window=window_size).std()
    return rolling_std

def plot_sequential_analysis(data, rolling_std):
    """Plot sequential analysis results."""
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data, label='Returns')
    plt.plot(rolling_std.index, rolling_std, label='Rolling Std Dev', color='orange')
    plt.title('Sequential Analysis')
    plt.xlabel('Time')
    plt.ylabel('Returns / Volatility')
    plt.legend()
    plt.show()

def main():
    # Generate synthetic data
    data = generate_synthetic_data()
    
    # Perform CUSUM analysis
    pos, neg = cusum(data['returns'])
    plot_cusum(data['returns'], pos, neg, threshold=1)
    
    # Perform Sequential Analysis
    rolling_std = sequential_analysis(data['returns'])
    plot_sequential_analysis(data['returns'], rolling_std)

if __name__ == "__main__":
    main()
