# Load and Preprocess Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('financial_data.csv')

# Assume data has a 'date' column and a 'price' column
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Calculate returns
data['returns'] = data['price'].pct_change().dropna()

# Standardize the data
scaler = StandardScaler()
data['returns_scaled'] = scaler.fit_transform(data['returns'].values.reshape(-1, 1))

# Feature Extraction
# Example: Use rolling window to calculate more features
window_size = 20
data['rolling_mean'] = data['returns_scaled'].rolling(window=window_size).mean()
data['rolling_std'] = data['returns_scaled'].rolling(window=window_size).std()

# Drop NaN values resulting from rolling calculation
data.dropna(inplace=True)

features = data[['returns_scaled', 'rolling_mean', 'rolling_std']]

# Train the GMM
gmm = GaussianMixture(n_components=3, random_state=42)
data['regime'] = gmm.fit_predict(features)

# Evaluate the Mode
# Evaluate the model by checking the regime probabilities
data['regime_prob'] = gmm.predict_proba(features).max(axis=1)

# Visualize the Results
# Plot the regimes
plt.figure(figsize=(14, 7))
for regime in range(gmm.n_components):
    regime_data = data[data['regime'] == regime]
    plt.plot(regime_data.index, regime_data['price'], label=f'Regime {regime}')

plt.title('Regimes over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Save and Load the Model
import joblib

# Save the model
joblib.dump(gmm, 'gmm_model.pkl')

# Load the model for future use
gmm = joblib.load('gmm_model.pkl')
























