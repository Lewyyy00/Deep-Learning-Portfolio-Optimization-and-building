
from src.data_preparation.data_fetcher import StockDataFetcher
from src.data_preparation.preprocess import preprocess_pipeline
from src.data_preparation.lstm_features import build_features, save_features
from src.data_preparation.preprocess import preprocess_pipeline
from src.markowitz_model.markowitz import markowitz_main
from src.lstm_model.lstm_training import lstm_pipeline
from src.lstm_model.lstm_tuning import tuning_main
from src.optimalization_model.cov_estimation import sigma_main
from src.optimalization_model.optimize_sharpe import optimalization_main
from src.backtest.backtest import backtest_main
from pathlib import Path
import numpy as np
import pandas as pd
import random
import tensorflow as tf

from src.config.project_variables import BACKTEST_SAVE_DIR, TICKERS, START_DATE, END_DATE




def clear_data_dirs(subdirs=None):

    base_path = Path("data")

    for i in subdirs:
        dir_path = base_path / i
        if not dir_path.exists():
            print(f"[INFO] Katalog nie istnieje: {dir_path}")
            continue

        for file in dir_path.glob("*"): # przeszukuje wszystkie pliki w katalogu
            if file.is_file(): # sprawdza czy to jest plik
                file.unlink() # usuwa plik
        print(f"Wyczyszczono: {dir_path}")



def main(data_cleanup=False):

    #DATA PIPELINE
    if data_cleanup:
        print("[0/8] Czyszczenie katalogów data/")
        clear_data_dirs(
            subdirs=["raw", "processed", "features", "predictions", "markowitz", 'backtests', 'tuning_Results']
        )
    print("\n[1/8] Pobranie surowych danych rynkowych dla analizowanych spółek z yFinance.")
    fetcher = StockDataFetcher(TICKERS, START_DATE, END_DATE)
    raw_data = fetcher.get_stock_data_yfinance()
    
    print("\n[2/8] Preprocessing (prices.csv, log_returns.csv)")
    aligned_data, prices, log_returns = preprocess_pipeline(raw_data)
    
    print("\n[3/8] Budowa zestawów cech wejściowych dla modeli LSTM i zapis ich w postaci plików features_{ticker}.csv")
    for ticker, df in aligned_data.items():
        features = build_features(df)
        save_features(features, ticker)

    #LSTM TRAINING + OPTIMALIZATION
    print("\n[4/8] Przeszukiwanie siatki hiperparametrów w celu wyboru najlepiej dopasowanych konfiguracji modeli.")
    tuning_main()
    
    print("\n[5/8] Uczenie sieci neuronowych i generowanie prognoz oczekiwanych stóp zwrotu.")
    lstm_pipeline()
    
    print("\n[6/8] Obliczenie macierzy kowariancji i wektora oczekiwanych stóp zwrotu na siatce dat rebalansowania portfela.")
    markowitz_main()
    sigma_main()
    
    print("\n[7/8] Wyznaczenie wag portfela dla modelu Markowitza oraz wariantu opartego na prognozach LSTM")
    optimalization_main(lstm=False)
    optimalization_main(lstm=True)

    # BACKTEST + METRYKI
    print("\n[8/8] Symulacja zachowania portfela w czasie oraz obliczenie miar efektywności inwestycyjnej.")
    backtest_main()
    print("\nPIPELINE ZAKOŃCZONY.")



def set_seed(seed):
    """
    Ustawia ziarno losowości.
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def summarize_metrics(df):
    """
    Liczy średnią i odchylenie standardowe dla każdej metryki w kolumnach df.
    """
    summary = pd.DataFrame({
        "mean": df.mean(numeric_only=True), #numeric_only=True aby uniknąć problemów z kolumnami nienumerycznymi
        "std": df.std(numeric_only=True, ddof=1) #ddof=1 to odchylenie standardowe próby 
    })
    return summary


def run_experiment_pipeline(n_runs=10, base_seed=100, recompute_sigma= False):
    

    if not recompute_sigma:
        print("[INFO] Estymacja sigma (raz)...")
        sigma_main()

    rows = []

    for i in range(n_runs):
        seed = base_seed + i
        print(f"\n=== RUN {i+1}/{n_runs} | seed={seed} ===")
        set_seed(seed)

        if recompute_sigma:
            sigma_main()

        lstm_pipeline()
        optimalization_main(lstm=True)
        bt = backtest_main()

        metrics_lstm = bt["lstm"]

        # dodaj metadane run
        metrics_lstm = dict(metrics_lstm)  
        metrics_lstm["run"] = i + 1
        metrics_lstm["seed"] = seed

        rows.append(metrics_lstm)

    
    results_df = pd.DataFrame(rows).sort_values("run").reset_index(drop=True)
    metric_cols = [c for c in results_df.columns if c not in ["run", "seed"]]
    summary_df = summarize_metrics(results_df[metric_cols])

    # Zapis
    out_dir = Path(BACKTEST_SAVE_DIR) / "repeated_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / f"lstm_runs_{n_runs}.csv"
    summary_path = out_dir / f"lstm_runs_{n_runs}_summary.csv"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path)

    print(f"\nZapisano wyniki: {results_path}")
    print(f"Zapisano podsumowanie: {summary_path}")

    return results_df, summary_df

if __name__ == "__main__":
    main(data_cleanup=False)
    run_experiment_pipeline(n_runs=10, base_seed=100, recompute_sigma=False)


