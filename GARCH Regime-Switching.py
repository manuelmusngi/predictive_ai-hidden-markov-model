import numpy as np
import pandas as pd
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
regimes = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
returns = np.zeros(n_samples)
for i in range(n_samples):
    if regimes[i] == 0:
        returns[i] = np.random.normal(0, 1)
    else:
        returns[i] = np.random.normal(0, 5)

# Fit Hidden Markov Model to detect regimes
hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000)
hmm.fit(returns.reshape(-1, 1))
hidden_states = hmm.predict(returns.reshape(-1, 1))

# Separate data by regimes
returns_regime_0 = returns[hidden_states == 0]
returns_regime_1 = returns[hidden_states == 1]

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
plt.plot(returns, label='Returns')
plt.plot(hidden_states, label='Regimes', linestyle='--')
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
