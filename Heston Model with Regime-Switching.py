import numpy as np
import matplotlib.pyplot as plt

# Model parameters
T = 1.0  # Time to maturity
N = 1000  # Number of time steps
dt = T / N  # Time step

# Heston model parameters for two regimes
kappa_1 = 2.0
theta_1 = 0.04
sigma_1 = 0.3
rho_1 = -0.7
v0_1 = 0.04

kappa_2 = 1.0
theta_2 = 0.02
sigma_2 = 0.2
rho_2 = -0.5
v0_2 = 0.02

# Markov chain transition matrix
transition_matrix = np.array([[0.95, 0.05],
                              [0.10, 0.90]])

def simulate_regime_switching(transition_matrix, N):
    regimes = np.zeros(N)
    regimes[0] = 0  # Start in regime 1
    for t in range(1, N):
        if regimes[t-1] == 0:
            regimes[t] = np.random.choice([0, 1], p=transition_matrix[0])
        else:
            regimes[t] = np.random.choice([0, 1], p=transition_matrix[1])
    return regimes

def simulate_heston_model(T, N, dt, kappa, theta, sigma, rho, v0):
    prices = np.zeros(N)
    volatilities = np.zeros(N)
    volatilities[0] = v0
    prices[0] = 100  # Initial price

    for t in range(1, N):
        dW1 = np.random.normal(0, np.sqrt(dt))
        dW2 = np.random.normal(0, np.sqrt(dt))
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW2

        volatilities[t] = volatilities[t-1] + kappa * (theta - volatilities[t-1]) * dt + sigma * np.sqrt(volatilities[t-1]) * dW2
        prices[t] = prices[t-1] * np.exp(-0.5 * volatilities[t-1] * dt + np.sqrt(volatilities[t-1] * dt) * dW1)

    return prices, volatilities

regimes = simulate_regime_switching(transition_matrix, N)
prices_1, volatilities_1 = simulate_heston_model(T, N, dt, kappa_1, theta_1, sigma_1, rho_1, v0_1)
prices_2, volatilities_2 = simulate_heston_model(T, N, dt, kappa_2, theta_2, sigma_2, rho_2, v0_2)

prices = np.zeros(N)
volatilities = np.zeros(N)
prices[0] = 100  # Initial price

for t in range(1, N):
    if regimes[t] == 0:
        prices[t] = prices_1[t]
        volatilities[t] = volatilities_1[t]
    else:
        prices[t] = prices_2[t]
        volatilities[t] = volatilities_2[t]

plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(prices, label="Asset Prices")
plt.title("Simulated Asset Prices with Regime-Switching")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(volatilities, label="Volatilities")
plt.title("Simulated Volatilities with Regime-Switching")
plt.legend()

plt.tight_layout()
plt.show()
