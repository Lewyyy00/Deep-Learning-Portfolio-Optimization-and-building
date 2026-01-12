import os
import numpy as np
import pandas as pd
from config.project_variables import WINDOW, FEATURES_SAVE_DIR


def sma(series, window):
    return series.rolling(window).mean()


def rolling_min(series, window):
    return series.rolling(window).min()


def rolling_max(series, window):
    return series.rolling(window).max()


def log_returns(price_series):
    return np.log(price_series / price_series.shift(1))


def momentum(log_ret, window):
    return log_ret.rolling(window).mean()


def volatility(log_ret, window):
    return log_ret.rolling(window).std()


def direction(log_ret):
    return (log_ret > 0).astype(int)


def rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def atr(high, low, close, window=14):
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()

    return atr

def build_features(df, windows=WINDOW):
    """
    df: DataFrame OHLCV + Adj_Close
    windows: list okien czasowych (np. [5, 10, 20])
    """

    features = pd.DataFrame(index=df.index)

    # ceny
    features["close"] = df["Adj_Close"]

    # log-zwroty
    features["log_return"] = log_returns(features["close"])

    # kierunek ruchu
    features["dir"] = direction(features["log_return"])

    # cechy kroczące
    for w in windows:
        features[f"sma_{w}"] = sma(features["close"], w)
        features[f"min_{w}"] = rolling_min(features["close"], w)
        features[f"max_{w}"] = rolling_max(features["close"], w)
        features[f"mom_{w}"] = momentum(features["log_return"], w)
        features[f"vol_{w}"] = volatility(features["log_return"], w)

    # RSI i ATR
    features["rsi_14"] = rsi(features["close"], 14)
    features["atr_14"] = atr(df["High"], df["Low"], df["Close"], 14)

    # target: przyszły log-zwrot (do ML)
    features["target"] = features["log_return"].shift(-1)

    # usunięcie NaN
    features = features.dropna()

    return features


def save_features(features, ticker, save_dir=FEATURES_SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    features.to_csv(f"{save_dir}/features_{ticker}.csv")
    print(f"Zapisano: {save_dir}/features_{ticker}.csv")

