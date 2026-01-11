from data_preparation.data_fetcher import StockDataFetcher
from data_preparation.preprocess import preprocess_pipeline
from data_preparation.lstm_features import build_features, save_features
from config.project_variables import *

"""TICKERS = ["AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA"]
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
"""

# 1. Pobranie danych
fetcher = StockDataFetcher(TICKERS, START_DATE, END_DATE)
raw_data = fetcher.get_stock_data_yfinance()

# 2. Preprocess (ujednolicenie dat)
aligned_data, prices, log_returns = preprocess_pipeline(raw_data)

# 3. Feature engineering
for ticker, df in aligned_data.items():
    features = build_features(df, windows=[5, 10, 20])
    save_features(features, ticker)