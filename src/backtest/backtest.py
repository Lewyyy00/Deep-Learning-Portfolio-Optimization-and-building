import numpy as np
import pandas as pd
from pathlib import Path

from src.config.project_variables import (
    ESTIMATION_WINDOW,
    REBALANCE_STEP,
    RETURNS_PATH,
    BACKTEST_SAVE_DIR,
    TICKERS,
    START_CAPITAL,             
    TEST_START_DATE,           
    END_DATE,                  
    TRADING_DAYS,              
    RISK_FREE_RATE_ANNUAL,     
    PRICES_PATH,       
    MARKOWITZ_SAVE_DIR,
    SEQ_LEN,
    BATCH_SIZE,
    EPOCHS        
)

from src.optimalization_model.optimize_sharpe import annual_to_daily_rf


def load_prices():
    """
    Wczytuje prices.csv w formacie:
    Date | AAPL | MSFT | ... | TSLA
    gdzie są ceny (Adj Close).
    """
    path = PRICES_PATH
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_weights(weights_path):
    """
    Wczytuje wagi:
    Date | tickery... | sharpe_daily
    """
    df = pd.read_csv(weights_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_drawdown(equity):
    """
    Drawdown_t = equity_t / max_{s<=t}(equity_s) - 1 
    """
    peak = equity.cummax() # maksymalne dotychczasowe equity 
    dd = equity / peak - 1.0
    return dd


def performance_metrics(portfolio_value):
    """
    Liczy metryki na podstawie dziennych log-zwrotów portfela.
    """
    value = portfolio_value.dropna()
    log_ret = np.log(value / value.shift(1)).dropna()

    mean_daily = log_ret.mean()
    vol_daily = log_ret.std()

    rf_daily = annual_to_daily_rf(RISK_FREE_RATE_ANNUAL, TRADING_DAYS)

    # Sharpe (roczny) – na log-zwrotach jako przybliżenie
    sharpe_annual = np.sqrt(TRADING_DAYS) * ((mean_daily - rf_daily) / (vol_daily + 1e-12))

    # Sortino: odchylenie tylko dla ujemnych zwrotów
    downside = log_ret[log_ret < 0]
    downside_std = downside.std()
    sortino_annual = np.sqrt(TRADING_DAYS) * ((mean_daily - rf_daily) / (downside_std + 1e-12))

    dd = compute_drawdown(value)
    max_dd = dd.min()

    sterling_ratio = (mean_daily * TRADING_DAYS) / (abs(max_dd) + 1e-12)

   #metryki efektywności portfela
    return {
        "start_balance": round(float(value.iloc[0]), 2),
        "end_balance": round(float(value.iloc[-1]), 2),
        "mean_daily_log_return": round(float(mean_daily), 6),
        "vol_daily_log_return": round(float(vol_daily), 6),
        "sharpe_annual": round(float(sharpe_annual), 4),
        "sortino_annual": round(float(sortino_annual), 4),
        "max_drawdown": round(float(max_dd), 4),
        "sterling_ratio": round(float(sterling_ratio), 6),
    }


def run_backtest(prices, weights, initial_capital):
    """
    Backtest z rebalansowaniem.
    Portfel jest rebalansowany w dniach z weights["Date"].
    Utrzymujemy stałą liczbę akcji między rebalansami.
    """
    prices = prices.copy()
    weights = weights.copy()

    # filtr okresu testowego
    start = pd.to_datetime(TEST_START_DATE)
    end = pd.to_datetime(END_DATE)

    prices = prices[(prices["Date"] >= start) & (prices["Date"] <= end)].reset_index(drop=True)

    # ustaw indeks na Date dla łatwych lookupów
    prices_idx = prices.set_index("Date")[TICKERS]

    rebalance_dates = weights["Date"].tolist()

    # kapitał i liczba akcji
    capital = initial_capital
    shares = pd.Series(0.0, index=TICKERS)

    # wynik dzienny
    out_rows = []

    for date in prices_idx.index:
        px = prices_idx.loc[date]

        # rebalans jeśli to dzień rebalansu
        if date in rebalance_dates:
            w_row = weights[weights["Date"] == date].iloc[0]
            w = w_row[TICKERS].values.astype(float)

            # wartość portfela tuż przed rebalansowaniem (po cenach z tego dnia)
            portfolio_value = float((shares * px).sum())
            if portfolio_value == 0.0:
                portfolio_value = capital  # pierwszy rebalans

            # nowe liczby akcji (ułamkowe) zgodnie z wagami
            target_dollars = portfolio_value * w
            shares = target_dollars / px.values

        # dzienna wycena
        value_today = float((shares * px).sum())

        if value_today <= 0.0:
            value_today = capital  # zabezpieczenie na wypadek zerowej wyceny

        out_rows.append({"Date": date, "PortfolioValue": value_today})

    out = pd.DataFrame(out_rows)

    prev = out["PortfolioValue"].shift(1)
    valid = (out["PortfolioValue"] > 0) & (prev > 0)

    out["LogReturn"] = np.nan
    out.loc[valid, "LogReturn"] = np.log(out.loc[valid, "PortfolioValue"] / prev.loc[valid])
    return out


def backtest_main():

    # stworzenie katalogu na wyniki backtestu
    BACKTEST_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_prices()

    # scieżki do wag
    weights_markowitz_path = MARKOWITZ_SAVE_DIR / f"weights_sharpe_markowitz_W{ESTIMATION_WINDOW}_step{REBALANCE_STEP}.csv"
    weights_lstm_path = MARKOWITZ_SAVE_DIR / f"weights_sharpe_lstm_seq{SEQ_LEN}_e{EPOCHS}_b{BATCH_SIZE}_step{REBALANCE_STEP}.csv"

    # wczytanie wag
    weights_markowitz = load_weights(weights_markowitz_path)
    weights_lstm = load_weights(weights_lstm_path)

    # backtesty
    backtest_markowitz = run_backtest(prices, weights_markowitz, START_CAPITAL) 
    backtest_lstm = run_backtest(prices, weights_lstm, START_CAPITAL)

    # metryki
    markowitz_metrics = performance_metrics(backtest_markowitz["PortfolioValue"])
    lstm_metrics = performance_metrics(backtest_lstm["PortfolioValue"])

    # zapis wyników
    backtest_markowitz.to_csv(BACKTEST_SAVE_DIR / f"equity_markowitz_W{ESTIMATION_WINDOW}_step{REBALANCE_STEP}.csv", index=False) #index=False żeby nie zapisywało indeksu DataFrame jako kolumny
    backtest_lstm.to_csv(BACKTEST_SAVE_DIR / f"equity_lstm_seq{SEQ_LEN}_e{EPOCHS}_b{BATCH_SIZE}_step{REBALANCE_STEP}.csv", index=False)

    print("=== Markowitz ===")
    for k, v in markowitz_metrics.items():
        print(f"{k}: {v}")

    print("\n=== LSTM ===")
    for k, v in lstm_metrics.items():
        print(f"{k}: {v}")

    print("\nZapisano serie wartości portfeli do:", BACKTEST_SAVE_DIR)

if __name__ == "__main__":
    backtest_main()    

