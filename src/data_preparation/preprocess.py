import pandas as pd
import numpy as np
import os

from data_fetcher import StockDataFetcher



REQUIRED_COLS = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]


def check_required_columns(df, ticker):
    """
    Sprawdza, czy dataframe ma wymagane kolumny OHLCV + Adj_Close.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: brakuje kolumn: {missing}. Dostępne: {list(df.columns)}")


def check_missing_values(df, ticker):
    """
    Sprawdza, czy są braki danych. Jeśli są, zwraca podsumowanie.
    """
    na_counts = df.isna().sum()
    total_na = int(na_counts.sum())
    if total_na > 0:
        print(f"[UWAGA] {ticker}: znaleziono braki danych (NaN).")
        print(na_counts[na_counts > 0])
    return total_na


def align_common_dates(data_dict):
    """
    Ujednolica daty: bierze przecięcie indeksów dat dla wszystkich tickerów.
    data_dict: dict {ticker: df}
    Zwraca nowy dict {ticker: df_aligned}
    """
    # przecięcie indeksów
    common_index = None
    for ticker, df in data_dict.items():
        idx = pd.to_datetime(df.index)
        common_index = idx if common_index is None else common_index.intersection(idx)

    common_index = common_index.sort_values()

    aligned = {}
    for ticker, df in data_dict.items():
        df2 = df.copy()
        df2.index = pd.to_datetime(df2.index)
        df2 = df2.sort_index()

        df2 = df2.loc[common_index]  # wybór wspólnych dat
        aligned[ticker] = df2

    return aligned


def build_prices_and_log_returns(aligned_dict):
    """
    Buduje macierz cen (Adj_Close) i macierz log-zwrotów.
    Zwraca: prices_df, log_returns_df
    """
    # prices: daty x tickery
    prices = pd.DataFrame({
        ticker: df["Adj_Close"] for ticker, df in aligned_dict.items()
    }).sort_index()

    # log-zwroty
    log_returns = np.log(prices / prices.shift(1)).dropna()

    return prices, log_returns


def save_processed_data(prices, log_returns, save_dir):
    """
    Zapisuje prices i log_returns do data/processed.
    """
    os.makedirs(save_dir, exist_ok=True)

    prices.to_csv(os.path.join(save_dir, "prices.csv"))
    log_returns.to_csv(os.path.join(save_dir, "log_returns.csv"))

    print(f"Zapisano: {save_dir}/prices.csv")
    print(f"Zapisano: {save_dir}/log_returns.csv")


def preprocess_pipeline(data_dict, save_dir="Deep-Learning-Portfolio-Optimization-and-building/data/processed"):
    """
    Główna funkcja: bierze surowe dane {ticker: df} i robi cały preprocess.
    Zwraca: aligned_dict, prices, log_returns
    """
    # 1) walidacja kolumn + braki
    for ticker, df in data_dict.items():
        check_required_columns(df, ticker)
        check_missing_values(df, ticker)

    # 2) ujednolicenie dat
    aligned = align_common_dates(data_dict)

    # 3) budowa cen i log-zwrotów
    prices, log_returns = build_prices_and_log_returns(aligned)

    # 4) zapis
    save_processed_data(prices, log_returns, save_dir=save_dir)

    return aligned, prices, log_returns



tickers = ["NVDA", "AAPL", "GOOG", "MSFT", "AMZN", "META", "TSLA"]
save_dir="Deep-Learning-Portfolio-Optimization-and-building/data/processed"

fetcher = StockDataFetcher(tickers, "2020-01-01", "2025-12-31")
raw_data = fetcher.get_stock_data_yfinance()   # dict: {ticker: df}
aligned, prices, log_returns = preprocess_pipeline(raw_data, save_dir)

