# Import Libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# Load and Preprocess Data
# Load your time series data
data = pd.read_csv('path_to_your_data.csv')

# Assume the time series is in a column called 'value'
ts = data['value']

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(ts)
plt.title('Time Series Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Define the SETAR Model
class SETAR:
    def __init__(self, ts, delay, regimes):
        self.ts = ts
        self.delay = delay
        self.regimes = regimes
        self.models = []

    def fit(self):
        for regime in self.regimes:
            model = MarkovRegression(self.ts, k_regimes=regime)
            result = model.fit()
            self.models.append(result)

    def predict(self, steps):
        forecasts = []
        for model in self.models:
            forecast = model.predict(start=len(self.ts), end=len(self.ts) + steps - 1)
            forecasts.append(forecast)
        return forecasts

# Estimate Model Parameters
# Define delay and number of regimes
delay = 1
regimes = [2, 3]  # Example with 2 and 3 regimes

# Initialize and fit the model
setar = SETAR(ts, delay, regimes)
setar.fit()


# Evaluate the Model
# Print summary of the models
for i, model in enumerate(setar.models):
    print(f"Model with {regimes[i]} regimes:")
    print(model.summary())

# Forecasting
# Forecast future values
steps = 10  # Number of steps to forecast
forecasts = setar.predict(steps)

# Plot forecasts
plt.figure(figsize=(10, 6))
plt.plot(ts, label='Original')
for i, forecast in enumerate(forecasts):
    plt.plot(range(len(ts), len(ts) + steps), forecast, label=f'Forecast {regimes[i]} regimes')
plt.title('SETAR Model Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()





