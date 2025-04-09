#### Regime Switching Models

Regime-switching models are widely used in financial markets to capture different market conditions, such as bull and bear markets, low and high volatility periods, or liquidity changes.

#### 1. Markov Regime-Switching Models
Hamilton Model (1989): One of the most popular regime-switching models, where financial data follows different regimes with transition probabilities governed by a Markov process.

Hidden Markov Models (HMMs): A probabilistic model where market states (regimes) are unobservable (hidden) and must be inferred from observed data such as returns or volatility.

Markov-based methodologies model the inherent uncertainty, regime shifts, and transitions in market behavior. They provide a probabilistic framework to not just understand historical trends but also to forecast future market conditions, making them invaluable tools in quantitative market analysis.

This quantitative analysis short study is a simplistic approach to the application of the hidden markov model on time series pattern recognition and market regime inferences that is amenable to systematic strategies implementation. Hidden Markov Models shows the ability to determine market regime state defined by any selected features.

  - [Hidden Markov Modeling - ES - E-mini S&P 500](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/1-Hidden-Markov-Modeling%20-%20ES%20-%20E-mini%20S%26P%20500.ipynb)
  - [Hidden-Markov-Models- ES - Systematic-Strategy](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/2-Hidden-Markov-Models-%20ES%20-%20Systematic-Strategy.ipynb)

#### 2. Stochastic Volatility with Regime-Switching Models
Heston Model with Regime-Switching: Volatility follows a stochastic process, but switches regimes based on a Markov chain.

GARCH with Regime-Switching: Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models with different volatility regimes.

#### 3. Machine Learning-Based Regime-Switching Models
Hidden Markov Model (HMM) with Bayesian Estimation: Uses machine learning techniques to estimate regime probabilities.

Recurrent Neural Networks (RNNs) & LSTMs: Capture regime shifts using time series memory mechanisms.

Gaussian Mixture Models (GMMs): Clusters financial time series into different regimes based on probability distributions.

#### 4. Threshold Autoregressive (TAR) Models
These models define regimes based on threshold levels of a variable (e.g., if volatility is above a certain threshold, it switches to a high-volatility regime).

Self-Exciting TAR (SETAR): Extends the TAR model by allowing for multiple regimes with different autoregressive processes.

#### 5. Smooth Transition Models
Smooth Transition Autoregressive (STAR) Models: Regimes change gradually rather than abruptly based on a logistic or exponential function.

Useful when market transitions are not sudden but evolve smoothly over time.

#### 6. Change-Point Detection Models
Bayesian Change-Point Models: Use Bayesian inference to detect structural breaks in market conditions.

CUSUM (Cumulative Sum) and Sequential Analysis: Identify shifts in mean returns or volatility over time.


##### Model References
  - [Detecting bearish and bullish markets in financial time series using hierarchical hidden Markov models](https://github.com/manuelmusngi/regime_switching_models/blob/main/2007.14874v1.pdf)
  - [Markov Models for Commodity Futures: Theory and Practice](https://github.com/manuelmusngi/regime_switching_models/blob/main/ssrn-1138782.pdf)
  - [Predicting Daily Probability Distributions of S&P500 Returns](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1288468)

##### Dependencies
  - [requirements.txt](https://github.com/manuelmusngi/hidden-markov-modeling/blob/main/requirements.txt)
  
  
