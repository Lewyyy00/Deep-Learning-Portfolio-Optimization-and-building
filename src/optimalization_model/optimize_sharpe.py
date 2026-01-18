import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.config.project_variables import (
    BATCH_SIZE,
    EPOCHS,
    SEQ_LEN,
    TEST_START_DATE,
    TICKERS,
    ESTIMATION_WINDOW,
    REBALANCE_STEP,
    MARKOWITZ_SAVE_DIR,
    RISK_FREE_RATE_ANNUAL,   
    TRADING_DAYS,
    LSTM_PREDICTION_SAVE_DIR            
)

from src.markowitz_model.markowitz import load_log_returns, get_rebalance_dates


def annual_to_daily_rf(r_annual, trading_days):
    """
    Konwersja rocznej stopy wolnej od ryzyka na dzienną (z kapitalizacją):
    r_d = (1 + r_a)^(1/252) - 1
    """
    return (1.0 + r_annual) ** (1.0 / trading_days) - 1.0


def load_mu_markowitz():
    """
    Wczytuje plik mu_W{W}_step{step}.csv
    Format: Date + kolumny tickers.
    """
    path = MARKOWITZ_SAVE_DIR / f"mu_markowitz_W{ESTIMATION_WINDOW}_step{REBALANCE_STEP}.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def load_mu_lstm():
    """
    Wczytuje plik mu_lstm_seq{SEQ_LEN}_e{EPOCHS}_b{BATCH_SIZE}.csv
    Format: Date + kolumny tickers.
    """
    path = MARKOWITZ_SAVE_DIR / f"mu_lstm_seq{SEQ_LEN}_e{EPOCHS}_b{BATCH_SIZE}.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_sigma_long():
    """
    Wczytuje plik sigma_W{W}_step{step}.csv w formacie long:
    Date, asset_i, asset_j, cov
    """
    path = MARKOWITZ_SAVE_DIR / f"sigma_W{ESTIMATION_WINDOW}_step{REBALANCE_STEP}.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def sigma_matrix_for_date(sigma_long, date):
    
    """
    Zwraca macierz kowariancji Sigma dla danej daty rebalansu.
    """
    
    sub = sigma_long[sigma_long["Date"] == date].copy()
    if sub.empty:
        raise ValueError(f"Brak danych Sigma dla daty {date.date()}")

    mat = sub.pivot(index="asset_i", columns="asset_j", values="cov") # pivot do macierzy, czyli rows: asset_i, cols: asset_j, values: cov
    mat = mat.loc[TICKERS, TICKERS]# upewniamy się, że kolejność jest zgodna z TICKERS

    return mat.values.astype(float)

def mu_vector_for_date(mu_df, date):

    """
    Zwraca wektor oczekiwanych zwrotów mu dla danej daty rebalansu.
    """

    row = mu_df[mu_df["Date"] == date]
    if row.empty:
        raise ValueError(f"Brak danych mu dla daty {date.date()} (wymagane, bo rebalans ma być w tym dniu).")
    return row.iloc[0][TICKERS].values.astype(float)


def sharpe_ratio(w, mu, sigma, rf_daily):
    """
    Sharpe (dzienny):
    S(w) = (w^T mu - rf) / sqrt(w^T Sigma w)
    """
    port_ret = float(w @ mu)
    port_var = float(w @ sigma @ w)
    port_vol = np.sqrt(max(port_var, 1e-12))  # ochrona przed 0

    return (port_ret - rf_daily) / port_vol


def optimize_max_sharpe(mu, sigma, rf_daily):
    """
    Maksymalizacja Sharpe’a przy:
    sum(w)=1, w_i>=0.
    Realizacja: minimalizacja -Sharpe (SLSQP).
    """
    n = len(mu)
    w0 = np.ones(n) / n # start: równe wagi
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}] # ograniczenia: suma wag = 1
    bounds = [(0.0, 1.0) for _ in range(n)] # granice: brak krótkiej sprzedaży

    # funkcja celu: -Sharpe, bo minimize() minimalizuje a nie maksymalizuje dlatego minus
    def objective(w):
        return -sharpe_ratio(w, mu, sigma, rf_daily)

    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons)

    if not res.success:
        raise RuntimeError(f"Optymalizacja nieudana: {res.message}")

    w_opt = np.clip(res.x, 0.0, 1.0) # mała korekta numeryczna (czasem suma wychodzi 0.9999999)
    w_opt = w_opt / w_opt.sum()

    return w_opt


def optimalization_main(lstm=False):
    rf_daily = annual_to_daily_rf(RISK_FREE_RATE_ANNUAL, TRADING_DAYS)

    dates = load_log_returns()

    rebalance_dates = get_rebalance_dates(
        df=dates,
        test_start_date=TEST_START_DATE,
        rebalance_step=REBALANCE_STEP,
        seq_len=SEQ_LEN
    )

    if lstm:
        mu_df = load_mu_lstm()
    else:
        mu_df = load_mu_markowitz()

    sigma_long = load_sigma_long()

    rows = []
    for date in rebalance_dates:
        mu = mu_vector_for_date(mu_df, date)
        sigma = sigma_matrix_for_date(sigma_long, date)
        w = optimize_max_sharpe(mu, sigma, rf_daily)

        out_row = {"Date": date}
        for i, t in enumerate(TICKERS):
            out_row[t] = float(w[i])

        # (opcjonalnie) diagnostyka Sharpe dla tej daty
        out_row["sharpe_daily"] = float(sharpe_ratio(w, mu, sigma, rf_daily))

        rows.append(out_row)

    w_df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

    if lstm:
        out_path = MARKOWITZ_SAVE_DIR / f"weights_sharpe_lstm_W{ESTIMATION_WINDOW}_step{REBALANCE_STEP}.csv"
    else:
        out_path = MARKOWITZ_SAVE_DIR / f"weights_sharpe_markowitz_W{ESTIMATION_WINDOW}_step{REBALANCE_STEP}.csv"

    w_df.to_csv(out_path, index=False)

    print(f"Zapisano: {out_path}")
    print("Liczba rebalansów:", len(w_df))

