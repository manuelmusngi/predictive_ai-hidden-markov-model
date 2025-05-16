#### Regime Switching Models
This project explores a variety of Regime-Switching Models to analyze financial time series data. These models aim to identify, predict, and understand market regime shifts, which are fundamental to improving trading strategies, risk management, and portfolio optimization.

The repository includes implementations of statistical, machine learning, and econometric models for regime-switching analysis, with a focus on production-ready pipelines for model training, evaluation, and backtesting.

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
├── src/\
│   ├── [__init__.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/init.py)          
│   ├── [data_loader.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/data_loader.py)        
│   ├── [feature_engineering.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/feature_engineering.py)  
│   ├── [hmm_model.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/hmm_model.py)          
│   ├── [signal_generation.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/signal_generation.py)  
│   ├── [backtesting.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/backtesting.py)        
│   ├── [utils.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/utils.py)              
│   ├── [plotter.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/plotter.py)             
├── [requirements.txt](https://github.com/manuelmusngi/regime_switching_models/blob/main/requirements.txt)          
├── [main.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/main.py)                   
└── [README.md](https://github.com/manuelmusngi/regime_switching_models/blob/main/README.md)  

#### Class-based development
- [regime_switching_system.py](https://github.com/manuelmusngi/regime_switching_models/blob/main/src/regime_switching_system.py)

#### Dependencies
  - [requirements.txt](https://github.com/manuelmusngi/hidden-markov-modeling/blob/main/requirements.txt)

#### Other Regime Switching Models

Further enhancements will include Other Regime Switching Models.
- [Other Regime Switching Models](https://github.com/manuelmusngi/regime_switching_models/blob/main/Other%20Regime%20Switching%20Models)

#### References
  - [Detecting bearish and bullish markets in financial time series using hierarchical hidden Markov models](https://github.com/manuelmusngi/regime_switching_models/blob/main/2007.14874v1.pdf)
  - [Markov Models for Commodity Futures: Theory and Practice](https://github.com/manuelmusngi/regime_switching_models/blob/main/ssrn-1138782.pdf)
  - [Predicting Daily Probability Distributions of S&P500 Returns](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1288468)
  - [Option Pricing in a Regime Switching Stochastic Volatility Model](https://arxiv.org/abs/1707.01237)
  - [A Markov-switching spatio-temporal ARCH model](https://arxiv.org/abs/2310.02630)

#### License
This project is licensed under the [MIT License](https://github.com/manuelmusngi/regime_switching_models/edit/main/LICENSE).


  
  
