import yfinance as yf
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt
import numpy as np


class StockDataFetcher:

    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None #pełne dane OHLCV
        self.adj_close = None


    def get_stock_data_yfinance(self):

        data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False,  #dodaje adj close
        )

        if data is None or data.empty:
            raise ValueError("nie udało się pobrać danych za pomocą yfinance.")

        self.data = data
        data.to_csv("stocks_data.csv")  
        return data


    def get_adj_close(self, data):

        adj_close = data["Adj Close"].dropna(how="any")
        self.adj_close = adj_close
        adj_close.to_csv("adj_close.csv")
        return adj_close

class indicators:

    def __init__(self):
      
        self.adj_close_path = "adj_close.csv"
        self.adj_close = self._load_adj_close()

        self.log_returns = None
        self.simple_returns = None

    def _load_adj_close(self):
     
        try:
            df = pd.read_csv(self.adj_close_path, index_col=0)
            df.index = pd.to_datetime(df.index)

        except FileNotFoundError:
            raise FileNotFoundError(f"Nie znaleziono pliku {self.adj_close_path}.")
        
        return df

    def compute_log_returns(self, save_csv=True, csv_path="log_returns.csv"):

        """
        Oblicza logarytmiczne stopy zwrotu: r_t = ln(P_t / P_{t-1}) na podstawie danych Adj Close.

        """
        
        log_ret = np.log(self.adj_close / self.adj_close.shift(1)).dropna()
        self.log_returns = log_ret

        if save_csv:
            log_ret.to_csv(csv_path)

        return log_ret
    
processor = indicators().compute_log_returns()
