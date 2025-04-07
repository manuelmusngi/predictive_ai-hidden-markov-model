# Load and Preprocess the Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic data with a change point
np.random.seed(42)
n = 100
change_point = 50
data = np.concatenate([np.random.normal(0, 1, change_point), np.random.normal(3, 1, n - change_point)])

# Convert to pandas DataFrame
df = pd.DataFrame(data, columns=['value'])

# Plot the data
plt.plot(df['value'])
plt.title('Synthetic Data with Change Point')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

#  Define the Bayesian Change-Point Model
import pymc3 as pm

with pm.Model() as model:
    # Priors for unknown model parameters
    mu1 = pm.Normal('mu1', mu=0, sigma=10)
    mu2 = pm.Normal('mu2', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Prior for the change point
    tau = pm.DiscreteUniform('tau', lower=0, upper=n)

    # Likelihood
    idx = np.arange(n)
    mu = pm.math.switch(tau >= idx, mu1, mu2)
    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=df['value'])

### Step 4: Perform Inference Using MCMC
Now, we sample from the posterior using MCMC.

```python
with model:
    trace = pm.sample(2000, tune=1000, cores=2, return_inferencedata=True)

# Plot the trace
pm.traceplot(trace)
plt.show()

# Summary of the trace
print(pm.summary(trace, var_names=['mu1', 'mu2', 'sigma', 'tau']))

# Analyze and Visualize the Results
# Posterior distribution of tau (change point)
plt.figure(figsize=(10, 5))
plt.hist(trace.posterior['tau'].values.flatten(), bins=n, alpha=0.75)
plt.axvline(change_point, color='r', linestyle='--', label='True Change Point')
plt.title('Posterior Distribution of Change Point')
plt.xlabel('Change Point')
plt.ylabel('Frequency')
plt.legend()
plt.show()

