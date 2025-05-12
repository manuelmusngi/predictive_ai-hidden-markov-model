import numpy as np
import yaml
import logging
import os
from datetime import datetime

# Logging Configuration
def configure_logging(log_file="logs/app.log"):
    """
    Configures the logging for the application.
    Logs messages to both the console and a file.
    """
    if not os.path.exists("logs/"):
        os.makedirs("logs/")
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging is configured.")

# Sharpe Ratio Calculation
def sharpe_ratio(returns_series, risk_free_rate=0.01, annual_trading_days=255):
    """
    Calculates the Sharpe Ratio for a given series of returns.

    Parameters:
    - returns_series (pd.Series or np.ndarray): Series of log returns.
    - risk_free_rate (float): Annualized risk-free rate (default: 0.01).
    - annual_trading_days (int): Number of trading days in a year (default: 255).

    Returns:
    - float: Sharpe ratio.
    """
    mean_return = returns_series.mean() * annual_trading_days
    std_dev = returns_series.std() * np.sqrt(annual_trading_days)
    sharpe = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else 0
    return round(sharpe, 2)

# YAML Configuration Loader
def load_config(config_path="config/config.yaml"):
    """
    Loads a YAML configuration file.

    Parameters:
    - config_path (str): Path to the configuration file.

    Returns:
    - dict: Configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Date and Time Utilities
def current_timestamp():
    """
    Returns the current timestamp in a readable format.

    Returns:
    - str: Current timestamp as `YYYY-MM-DD HH:MM:SS`.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
