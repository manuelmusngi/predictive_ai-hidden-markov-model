import numpy as np
import pandas as pd
from hmmlearn import hmm
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt

# Step 1: Generate Synthetic Data
np.random.seed(42)

# Parameters
n_states = 2
n_samples = 1000

# Transition matrix
transmat = np.array([[0.7, 0.3],
                     [0.3, 0.7]])

# Means and variances for each state
means = [0, 3]
variances = [1, 1]

# Generate synthetic data
states = np.zeros(n_samples, dtype=int)
observations = np.zeros(n_samples)

states[0] = np.random.choice(n_states)
observations[0] = np.random.normal(means[states[0]], np.sqrt(variances[states[0]]))

for t in range(1, n_samples):
    states[t] = np.random.choice(n_states, p=transmat[states[t-1]])
    observations[t] = np.random.normal(means[states[t]], np.sqrt(variances[states[t]]))

data = pd.DataFrame({'state': states, 'observation': observations})

# Step 2: Fit HMM Model
model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
model.fit(data['observation'].values.reshape(-1, 1))

# Predict the hidden states
hidden_states = model.predict(data['observation'].values.reshape(-1, 1))
data['hidden_state'] = hidden_states

# Step 3: Bayesian Estimation of Regime Probabilities
with pm.Model() as bayesian_model:
    # Priors for transition probabilities
    p_0_0 = pm.Beta('p_0_0', alpha=1, beta=1)
    p_0_1 = 1 - p_0_0
    p_1_0 = pm.Beta('p_1_0', alpha=1, beta=1)
    p_1_1 = 1 - p_1_0
    
    # Priors for means and variances
    mu_0 = pm.Normal('mu_0', mu=0, sigma=10)
    sigma_0 = pm.HalfNormal('sigma_0', sigma=1)
    mu_1 = pm.Normal('mu_1', mu=0, sigma=10)
    sigma_1 = pm.HalfNormal('sigma_1', sigma=1)
    
    # Observations
    obs = data['observation'].values
    
    def logp(s):
        return tt.switch(s,
                         pm.Normal.dist(mu=mu_1, sigma=sigma_1).logp(obs),
                         pm.Normal.dist(mu=mu_0, sigma=sigma_0).logp(obs))
    
    # Hidden states
    states = pm.Categorical('states', p=[p_0_0, p_0_1], shape=n_samples)
    
    # Likelihood
    likelihood = pm.Potential('likelihood', logp(states))
    
    # Sampling
    trace = pm.sample(1000, tune=500, cores=1)

# Step 4: Analyze the Results
# Plot the observations and the hidden states
plt.figure(figsize=(15, 5))
plt.plot(data['observation'], label='Observations')
plt.plot(data['hidden_state'], label='Hidden States', alpha=0.7)
plt.legend()
plt.show()

# Plot the posterior distributions
pm.plot_posterior(trace, var_names=['p_0_0', 'p_1_0', 'mu_0', 'mu_1', 'sigma_0', 'sigma_1'])
plt.show()
