import os
import yfinance as yf
import pandas as pd
from configparser import ConfigParser

class DataLoader:
    def __init__(self, symbol, start_date, end_date, save_path):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.save_path = save_path

    def fetch_data(self):
        data = yf.download(self.symbol, self.start_date, self.end_date)
        data.to_csv(os.path.join(self.save_path, f"{self.symbol}.csv"))
        return data

    def load_data(self):
        file_path = os.path.join(self.save_path, f"{self.symbol}.csv")
        if not os.path.exists(file_path):
            return self.fetch_data()
        return pd.read_csv(file_path, index_col="Date", parse_dates=True)
