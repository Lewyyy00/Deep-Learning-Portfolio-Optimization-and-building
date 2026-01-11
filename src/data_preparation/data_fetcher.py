import yfinance as yf
import pandas as pd
import numpy as np
import os

class StockDataFetcher:

    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}   # dane OHLCV per ticker
        self.save_dir = "Deep-Learning-Portfolio-Optimization-and-building/data/raw"

    def get_stock_data_yfinance(self):

        os.makedirs(self.save_dir, exist_ok=True)
        for ticker in self.tickers:

            df = yf.download(
                str(ticker),
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False, #dodaje adj close
                progress=False #wyłącza pasek postępu
            )

            if df is None or df.empty:
                raise ValueError(f"Nie udało się pobrać danych dla {ticker}")

            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            #obsługa danych MultiIndex
            if isinstance(df.columns[0], tuple):
                df.columns = [column[0].replace(" ", "_") for column in df.columns]
            else:
                df.columns = [column.replace(" ", "_") for column in df.columns]

            df.to_csv(f"{self.save_dir}/{ticker}.csv")
            self.data[ticker] = df

        return self.data

if __name__ == "__main__":
    
    tickers = ["NVDA", "AAPL", "GOOG", "MSFT", "AMZN", "META", "TSLA"]

    fetcher = StockDataFetcher(tickers, "2020-01-01", "2025-12-31")
    data = fetcher.get_stock_data_yfinance()  

    prices = pd.DataFrame({
        t: df["Adj_Close"] for t, df in data.items()
    })

    log_returns = np.log(prices / prices.shift(1)).dropna()