from src.data_preparation.data_fetcher import StockDataFetcher
from src.data_preparation.preprocess import preprocess_pipeline
from src.data_preparation.lstm_features import build_features, save_features
from src.config.project_variables import TICKERS, START_DATE, END_DATE

# 1. Pobranie danych z yFinance.
fetcher = StockDataFetcher(TICKERS, START_DATE, END_DATE)
raw_data = fetcher.get_stock_data_yfinance()

# 2. Preprocess, ktory obejmuje ujednolicenie dat, stworzenie plików .csv z cenami oraz logarytmicznych zwrotow.
aligned_data, prices, log_returns = preprocess_pipeline(raw_data)

# 3. Feature engineering - stworzenie cech dla LSTM i zapis do plików.
for ticker, df in aligned_data.items():
    features = build_features(df)
    save_features(features, ticker)
