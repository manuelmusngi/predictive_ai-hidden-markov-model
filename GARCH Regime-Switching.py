import numpy as np
import pandas as pd
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# Load SPX data
data = pd.read_csv('path_to_your_spx_data.csv')  # Replace with the path to your data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Calculate returns
data['Returns'] = data['Close'].pct_change().dropna()

# Prepare data for HMM
returns = data['Returns'].values[1:].reshape(-1, 1)

# Fit Hidden Markov Model to detect regimes
hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000)
hmm.fit(returns)
hidden_states = hmm.predict(returns)

# Add hidden states to the data
data = data.iloc[1:]  # Align with returns
data['Regime'] = hidden_states

# Separate data by regimes
returns_regime_0 = data.loc[data['Regime'] == 0, 'Returns']
returns_regime_1 = data.loc[data['Regime'] == 1, 'Returns']

# Fit GARCH models for each regime
garch_0 = arch_model(returns_regime_0, vol='Garch', p=1, q=1)
garch_1 = arch_model(returns_regime_1, vol='Garch', p=1, q=1)

res_garch_0 = garch_0.fit(disp="off")
res_garch_1 = garch_1.fit(disp="off")

print("GARCH Model for Regime 0:")
print(res_garch_0.summary())

print("\nGARCH Model for Regime 1:")
print(res_garch_1.summary())

# Plot results
plt.figure(figsize=(12, 8))
plt.plot(data['Returns'], label='Returns')
plt.plot(data['Regime'], label='Regimes', linestyle='--')
plt.legend()
plt.title('Returns and Detected Regimes')
plt.show()

# Forecast volatility for each regime
forecast_0 = res_garch_0.forecast(horizon=5)
forecast_1 = res_garch_1.forecast(horizon=5)

print("\nForecast Volatility for Regime 0:")
print(forecast_0.variance[-1:])

print("\nForecast Volatility for Regime 1:")
print(forecast_1.variance[-1:])
