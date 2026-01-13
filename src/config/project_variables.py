TICKERS = ["AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA"]
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
WINDOW = [5]
REQUIRED_COLS = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]

RAW_SAVE_DIR="data/raw"
PROCESSED_SAVE_DIR="data/processed"
FEATURES_SAVE_DIR="data/features"

FEATURE_COLS = ["close", "log_return", "dir", "mom_5", "vol_5", "rsi_14", "atr_14"]
TARGET_COL = "target"
SEQ_LEN = 20
BATCH_SIZE = 32