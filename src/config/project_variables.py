from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TICKERS = ["AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA"]
START_DATE = "2020-01-01"
TRAINING_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
WINDOW = [5]
REQUIRED_COLS = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]

RAW_SAVE_DIR="data/raw"
PROCESSED_SAVE_DIR="data/processed"
FEATURES_SAVE_DIR="data/features"
LSTM_PREDICTION_SAVE_DIR = PROJECT_ROOT / "data" / "predictions"
LSTM_FEATURES_SAVE_DIR = PROJECT_ROOT / "data" / "features"

FEATURE_COLS = ["close", "log_return", "dir", "mom_5", "vol_5", "rsi_14", "atr_14"]
TARGET_COL = "target"
SEQ_LEN = 20 #długość sekwencji wejściowej dla LSTM
BATCH_SIZE = 32 #rozmiar batcha podczas trenowania LSTM, im większy tym szybciej, ale wymaga więcej pamięci
TICKER = "AAPL"
EPOCHS = 10 #liczba epok trenowania LSTM