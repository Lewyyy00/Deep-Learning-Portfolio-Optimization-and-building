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
        
        data = data.reset_index()
        data.rename(columns={"Date": "Date"}, inplace=True)

        self.data = data
        data.to_csv(f"stocks_data_{self.tickers}.csv")  
        return data


if __name__ == "__main__":

    tickers = ["NVDA", "AAPL", "GOOG", "MSFT", "AMZN", "META", "TSLA"]
    start_date = "2020-01-01"
    end_date = "2025-12-31"

    for ticker in tickers:

        fetcher = StockDataFetcher(ticker, start_date, end_date)
        data = fetcher.get_stock_data_yfinance()
