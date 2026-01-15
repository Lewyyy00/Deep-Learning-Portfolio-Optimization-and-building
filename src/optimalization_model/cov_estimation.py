import pandas as pd
from pathlib import Path

from src.markowitz_model.markowitz import get_rebalance_dates, load_log_returns, get_history_window
from src.config.project_variables import (
    TICKERS,  
    ESTIMATION_WINDOW,   
    REBALANCE_STEP,
    MARKOWITZ_SAVE_DIR    
)


def estimate_sigma(hist):
    """
    Estymacja macierzy kowariancji Markowitza:
    Sigma_hat = cov(historycznych log-zwrotów) dla aktywów z listy TICKERS.
    """
    sigma_hat = hist[TICKERS].cov()
    return sigma_hat


def main():
    
    df = load_log_returns()
    rebalance_dates = get_rebalance_dates(df)

    # lista wyników w formacie long: Date, asset_i, asset_j, cov
    sigma_rows = []
    for date in rebalance_dates:
        hist = get_history_window(df, date, ESTIMATION_WINDOW)
        sigma_hat = estimate_sigma(hist)

        for i in TICKERS:
            for j in TICKERS:
                sigma_rows.append({
                    "Date": date,
                    "asset_i": i,
                    "asset_j": j,
                    "cov": float(sigma_hat.loc[i, j])
                })

    sigma_df = (
        pd.DataFrame(sigma_rows)
        .sort_values(["Date", "asset_i", "asset_j"])
        .reset_index(drop=True)
    )

    MARKOWITZ_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    sigma_path = MARKOWITZ_SAVE_DIR / f"sigma_W{ESTIMATION_WINDOW}_step{REBALANCE_STEP}.csv"
    sigma_df.to_csv(sigma_path, index=False)

    print(f"Zapisano: {sigma_path}")
