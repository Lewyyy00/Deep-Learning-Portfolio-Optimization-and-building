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
    SEQ_LEN
)

MARKOWITZ_SAVE_DIR.mkdir(parents=True, exist_ok=True) 


def load_log_returns():
    if not RETURNS_PATH.exists():
        raise FileNotFoundError(f"Brak pliku: {RETURNS_PATH}")

    df = pd.read_csv(RETURNS_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df

def get_rebalance_dates(df,
                        test_start_date,
                        rebalance_step,
                        seq_len):
    """
    Rebalans co 'rebalance_step' sesji, ale pierwszy rebalans dopiero wtedy,
    gdy LSTM może wygenerować pierwszą prognozę (seq_len - 1 sesji po starcie testu).

    df: DataFrame z kolumną Date (posortowany)
    """
    df = df.sort_values("Date").reset_index(drop=True)
    test_start = pd.to_datetime(test_start_date)

    test_dates = df[df["Date"] >= test_start]["Date"].tolist() # wszystkie sesje testowe od startu testu
    start_idx = seq_len - 1 # przesunięcie startu: musimy mieć seq_len obserwacji, więc start = seq_len-1
    rebalance_dates = test_dates[start_idx::rebalance_step] # daty rebalansu: startując od start_idx, co rebalance_step
    
    return rebalance_dates

def get_history_window(df, rebalance_date, window):
    """
    Zwraca ostatnie 'window' sesji PRZED datą rebalansu (bez look-ahead).
    """
    hist = df[df["Date"] < rebalance_date].tail(window) #tail zwraca ostatnie 'window' wierszy

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


def markowitz_main():
    df = load_log_returns()

    # daty rebalansu w 2025
    rebalance_dates = get_rebalance_dates(
        df=df,
        test_start_date=TEST_START_DATE,
        rebalance_step=REBALANCE_STEP,
        seq_len=SEQ_LEN
)


    rows = []
    for date in rebalance_dates:
        mu_hat = estimate_mu(df, date, ESTIMATION_WINDOW)
        row = {"Date": date}
        for t in TICKERS:
            row[t] = mu_hat[t]
        rows.append(row)

    mu_df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

    save_dir = MARKOWITZ_SAVE_DIR / f"mu_markowitz_W{ESTIMATION_WINDOW}_step{REBALANCE_STEP}.csv"
    mu_df.to_csv(save_dir, index=False)

    print(f"Zapisano: {save_dir}")
   
if __name__ == "__main__":
    markowitz_main()