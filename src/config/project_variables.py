from pathlib import Path

TICKERS = ["AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA"] #lista tickerów z MAG7 do analizy
START_DATE = "2020-01-01" #data rozpoczęcia analizy
TRAINING_END_DATE = "2024-12-31" #data zakończenia treningu LSTM
TEST_START_DATE = "2025-01-01" #data rozpoczęcia testu LSTM
END_DATE = "2025-12-31" #data zakończenia analizy
WINDOW = [5] #okresy do obliczania cech technicznych
REQUIRED_COLS = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"] #wymagane kolumny w danych surowych

# Ścieżki do zapisu danych
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_SAVE_DIR="data/raw"
PROCESSED_SAVE_DIR="data/processed"
FEATURES_SAVE_DIR="data/features"
LSTM_PREDICTION_SAVE_DIR = PROJECT_ROOT / "data" / "predictions"
LSTM_FEATURES_SAVE_DIR = PROJECT_ROOT / "data" / "features"
MARKOWITZ_SAVE_DIR = PROJECT_ROOT / "data" / "markowitz"
RETURNS_PATH = PROJECT_ROOT / "data" / "processed" / "log_returns.csv"
PRICES_PATH = PROJECT_ROOT / "data" / "processed" / "prices.csv"

FEATURE_COLS = ["close", "log_return", "dir", "mom_5", "vol_5", "rsi_14", "atr_14"] #kolumny cech dla LSTM
TARGET_COL = "target" #zmienna docelowa dla LSTM
SEQ_LEN = 20 #długość sekwencji wejściowej dla LSTM
BATCH_SIZE = 32 #rozmiar batcha podczas trenowania LSTM, im większy tym szybciej, ale wymaga więcej pamięci
TICKER = "AAPL"
EPOCHS = 10 #liczba epok trenowania LSTM

ESTIMATION_WINDOW = 252 #okres estymacji kowariancji (dni handlowe w roku)
REBALANCE_STEP = 20 #co ile dni rebalans portfela
RISK_FREE_RATE_ANNUAL = 0.043 #roczna stopa wolna od ryzyka 
TRADING_DAYS = 252 #liczba dni handlowych w roku
START_CAPITAL = 100000 #początkowy kapitał do backtestu