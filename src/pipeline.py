
from src.data_preparation.data_fetcher import StockDataFetcher
from src.data_preparation.preprocess import preprocess_pipeline
from src.data_preparation.lstm_features import build_features, save_features
from src.data_preparation.preprocess import preprocess_pipeline
from src.lstm_model.lstm_training import lstm_pipeline
from src.optimalization_model.cov_estimation import sigma_main
from src.optimalization_model.optimize_sharpe import optimalization_main
from src.backtest.backtest import backtest_main

from src.config.project_variables import TICKERS, START_DATE, END_DATE

def main():

    #DATA PIPELINE
    print("\n[1/7] Pobieranie danych z yFinance")
    fetcher = StockDataFetcher(TICKERS, START_DATE, END_DATE)
    raw_data = fetcher.get_stock_data_yfinance()
    print("\n[2/7] Preprocessing (prices.csv, log_returns.csv)")
    aligned_data, prices, log_returns = preprocess_pipeline(raw_data)
    print("\n[3/7] Feature engineering (features_*.csv)")
    for ticker, df in aligned_data.items():
        features = build_features(df)
        save_features(features, ticker)

    #LSTM TRAINING + OPTIMALIZATION
    print("\n[4/7] Budowa mu_lstm na siatce rebalansów i LSTM trening + predykcje (lstm_pred_*.csv)")
    lstm_pipeline()
    print("\n[5/7] Estymacja sigma na siatce rebalansów ")
    sigma_main()
    print("\n[6/7] Optymalizacja Sharpe (Markowitz i LSTM)")
    optimalization_main(lstm=False)
    optimalization_main(lstm=True)

    # BACKTEST + METRYKI
    print("\n[7/7] Backtest + metryki")
    backtest_main()
    print("\nPIPELINE ZAKOŃCZONY.")

def run_experiment_pipeline():
    
    pass


if __name__ == "__main__":
    main()
