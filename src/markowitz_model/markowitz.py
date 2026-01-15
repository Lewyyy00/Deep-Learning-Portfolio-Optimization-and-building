import pandas as pd

from src.config.project_variables import (
    TICKERS,
    TRAINING_END_DATE,
    TEST_START_DATE,
    ESTIMATION_WINDOW,
    REBALANCE_STEP,
    ESTIMATION_WINDOW,
    REBALANCE_STEP,
    MARKOWITZ_SAVE_DIR,
    RETURNS_PATH,
)

MARKOWITZ_SAVE_DIR.mkdir(parents=True, exist_ok=True) 


def load_log_returns():
    if not RETURNS_PATH.exists():
        raise FileNotFoundError(f"Brak pliku: {RETURNS_PATH}")

    df = pd.read_csv(RETURNS_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def get_rebalance_dates(df):
    """
    Daty rebalansu bierzemy z okresu testowego (2025),
    co REBALANCE_STEP sesji.
    """
    test_df = df[df["Date"] >= pd.to_datetime(TEST_START_DATE)].copy()
    dates = test_df["Date"].tolist()

    # co k-ty dzień w sensie sesji giełdowej (indeksowej), nie kalendarzowo
    rebalance_dates = dates[::REBALANCE_STEP]
    return rebalance_dates

def get_history_window(df, rebalance_date, window):
    """
    Zwraca ostatnie 'window' sesji PRZED datą rebalansu (bez look-ahead).
    """
    hist = df[df["Date"] < rebalance_date].tail(window)

    if len(hist) < window:
        raise ValueError(
            f"Za mało danych historycznych przed {rebalance_date.date()}: "
            f"jest {len(hist)}, potrzeba {window}."
        )
    return hist


def estimate_mu(df, rebalance_date, window):
    """
    Klasyczna estymacja Markowitza:
    mu_hat = średnia historycznych log-zwrotów z ostatnich 'window' sesji
    dostępnych PRZED datą rebalansu (bez look-ahead).
    """
    hist = get_history_window(df, rebalance_date, window)
    mu_hat = hist[TICKERS].mean(axis=0)
    return mu_hat


def main():
    df = load_log_returns()

    # daty rebalansu w 2025
    reb_dates = get_rebalance_dates(df)

    rows = []
    for date in reb_dates:
        mu_hat = estimate_mu(df, date, ESTIMATION_WINDOW)
        row = {"Date": date}
        for t in TICKERS:
            row[t] = mu_hat[t]
        rows.append(row)

    mu_df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

    save_dir = MARKOWITZ_SAVE_DIR / f"mu_markowitz_W{ESTIMATION_WINDOW}_step{REBALANCE_STEP}.csv"
    mu_df.to_csv(save_dir, index=False)

    print(f"Zapisano: {save_dir}")
   
