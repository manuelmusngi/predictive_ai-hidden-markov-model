#### Regime Switching Models

Regime-switching models are widely used in financial markets to capture different market conditions, such as bull and bear markets, low and high volatility periods, or liquidity changes.

#### Markov Regime-Switching Models
Markov-based methodologies model the inherent uncertainty, regime shifts, and transitions in market behavior. They provide a probabilistic framework to not just understand historical trends but also to forecast future market conditions, making them invaluable tools in quantitative market analysis.


Hidden Markov Models (HMMs): A probabilistic model where market states (regimes) are unobservable (hidden) and must be inferred from observed data such as returns or volatility.

This quantitative analysis short study is a simplistic approach to the application of the hidden markov model on time series pattern recognition and market regime inferences that is amenable to systematic strategies implementation. Hidden Markov Models shows the ability to determine market regime state defined by any selected features.

  - [Hidden Markov Modeling - ES - E-mini S&P 500](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/1-Hidden-Markov-Modeling-%20ES%20-%20E-mini%20S%26P%20500.ipynb)
  - [Hidden-Markov-Models- ES - Systematic-Strategy](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/2-Hidden-Markov-Models-%20ES%20-%20Systematic-Strategy.ipynb)

#### Project directory structure  

regime_switching_models/\
├── config/\
│   ├── [config.yaml](https://github.com/manuelmusngi/regime_switching_models/blob/main/config/config.yaml)\
├── data/\
│   ├── raw/                  
│   ├── processed/            
├── logs/\
│   ├── app.log               
├── src/\
│   ├── [__init__.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/init.py)          
│   ├── [data_loader.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/data_loader.py)        
│   ├── [feature_engineering.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/feature_engineering.py)  
│   ├── [hmm_model.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/hmm_model.py)          
│   ├── [signal_generation.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/signal_generation.py)  
│   ├── [backtesting.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/backtesting.py)        
│   ├── [utils.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/utils.py)              
│   ├── [plotter.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/plotter.py)            
├── tests/\
│   ├── test_data_loader.py  
│   ├── test_feature_engineering.py  
│   ├── test_hmm_model.py     
│   ├── test_signal_generation.py  
│   ├── test_backtesting.py   
├── notebooks/\
│   ├── exploratory_analysis.ipynb  
├── requirements.txt          
├── main.py                   
├── README.md                 
└── setup.py                  

#### Dependencies
  - [requirements.txt](https://github.com/manuelmusngi/hidden-markov-modeling/blob/main/requirements.txt)

#### Other Regime Switching Models

[Other Regime Switching Models](https://github.com/manuelmusngi/regime_switching_models/blob/main/Other%20Regime%20Switching%20Models)

#### Model References
  - [Detecting bearish and bullish markets in financial time series using hierarchical hidden Markov models](https://github.com/manuelmusngi/regime_switching_models/blob/main/2007.14874v1.pdf)
  - [Markov Models for Commodity Futures: Theory and Practice](https://github.com/manuelmusngi/regime_switching_models/blob/main/ssrn-1138782.pdf)
  - [Predicting Daily Probability Distributions of S&P500 Returns](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1288468)
  - [Option Pricing in a Regime Switching Stochastic Volatility Model](https://arxiv.org/abs/1707.01237)
  - [A Markov-switching spatio-temporal ARCH model](https://arxiv.org/abs/2310.02630)


  
  
