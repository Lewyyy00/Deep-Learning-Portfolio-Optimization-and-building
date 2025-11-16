import yfinance as yf
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt


class StockDataFetcher:

    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
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

