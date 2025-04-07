import numpy as np
import pandas as pd

class RegimeSwitchingModel:
    def __init__(self, transition_matrix, regimes):
        """
        Initialize the Regime Switching Model.

        Parameters:
        - transition_matrix: A 2D numpy array representing the state transition probabilities.
        - regimes: A list of functions representing different regimes.
        """
        self.transition_matrix = transition_matrix
        self.regimes = regimes
        self.current_state = 0

    def switch_regime(self):
        """
        Switch to the next regime based on the transition probabilities.
        """
        self.current_state = np.random.choice(
            len(self.transition_matrix),
            p=self.transition_matrix[self.current_state]
        )
        return self.current_state

    def simulate(self, data, steps):
        """
        Simulate the regime-switching model for a given number of steps.

        Parameters:
        - data: A pandas DataFrame with columns date, open, close, high, low, and volume.
        - steps: Number of steps to simulate.

        Returns:
        - A DataFrame with the original data and an additional column for regime values.
        """
        results = []
        for _ in range(steps):
            regime_function = self.regimes[self.current_state]
            result = regime_function(data)
            results.append(result)
            self.switch_regime()
        
        regime_column = pd.Series(results, name="regime")
        return pd.concat([data.reset_index(drop=True), regime_column], axis=1)

#usage:

# Define two example regimes
def regime_1(data):
    return np.random.choice(data['close'])  # Randomly choose a close value from the data

def regime_2(data):
    return np.random.choice(data['high'])  # Randomly choose a high value from the data

# Define transition matrix
transition_matrix = np.array([
    [0.9, 0.1],  # 90% chance of staying in regime 1, 10% chance of switching to regime 2
    [0.2, 0.8]   # 20% chance of switching to regime 1, 80% chance of staying in regime 2
])

# Create sample data
data = pd.DataFrame({
    'date': pd.date_range(start='2022-01-01', periods=100, freq='D'),
    'open': np.random.rand(100) * 100,
    'close': np.random.rand(100) * 100,
    'high': np.random.rand(100) * 100,
    'low': np.random.rand(100) * 100,
    'volume': np.random.randint(1000, 10000, size=100)
})

# Create the regime-switching model
model = RegimeSwitchingModel(transition_matrix, [regime_1, regime_2])

# Simulate the model for the length of the data
simulation_results = model.simulate(data, len(data))

# Print the results
print(simulation_results)
