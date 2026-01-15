import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.config.project_variables import (
    FEATURE_COLS, TARGET_COL, SEQ_LEN, BATCH_SIZE,
    TRAINING_END_DATE, TEST_START_DATE,
    EPOCHS, LSTM_PREDICTION_SAVE_DIR, LSTM_FEATURES_SAVE_DIR,
    TICKERS
)

def load_data(ticker):
    """
    Wczytuje plik features_{ticker}.csv, konwertuje Date i sortuje dane rosnąco po czasie.
    """
    path = LSTM_FEATURES_SAVE_DIR / f"features_{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    df = pd.read_csv(path)

    if "Date" not in df.columns:
        raise ValueError("Brakuje kolumny 'Date' w pliku features.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def split_train_test(df,
                     train_end=TRAINING_END_DATE,
                     test_start=TEST_START_DATE):
    """
    Podział czasowy (walidacja out-of-sample): train=2020–2024, test=2025.
    Usuwa wiersze z brakami w cechach i w zmiennej objaśnianej.
    """
    train_df = df[df["Date"] <= train_end].dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    test_df = df[df["Date"] >= test_start].dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    return train_df, test_df


def make_dataset(X, y, shuffle):
    """
    Buduje dataset sekwencyjny dla LSTM.
    Każda próbka to okno długości SEQ_LEN; target odpowiada końcowi okna.
    """
    return tf.keras.utils.timeseries_dataset_from_array(
        data=X,
        targets=y,
        sequence_length=SEQ_LEN,
        sequence_stride=1,
        shuffle=shuffle,
        batch_size=BATCH_SIZE,
    )


def build_model(n_features):
    """
    Buduje model LSTM do regresji zwrotów.
    32 jednostki LSTM, 1 neuron wyjściowy.
    Wykorzystuje optymalizator Adam i funkcje straty MSE.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQ_LEN, n_features)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def save_predictions(pred_df: pd.DataFrame, ticker: str):
    """
    Zapisuje prognozy modelu LSTM do data/predictions/ jako lstm_pred_{ticker}.csv.
    """
    LSTM_PREDICTION_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    path = LSTM_PREDICTION_SAVE_DIR / f"lstm_pred_{ticker}.csv"
    pred_df.to_csv(path, index=False)
    print(f"[{ticker}] Zapisano: {path}")

def train_for_ticker(ticker):
    df = load_data(ticker)
    train_df, test_df = split_train_test(df)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_ds = make_dataset(X_train, y_train, shuffle=True)
    test_ds = make_dataset(X_test, y_test, shuffle=False)

    model = build_model(n_features=len(FEATURE_COLS))
    model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds, verbose=1)

    preds = model.predict(test_ds, verbose=0).flatten() # spłaszczanie do 1D (liczba_sekwencji, 1) --> (liczba_sekwencji,)
    dates = test_df["Date"].iloc[SEQ_LEN - 1:].reset_index(drop=True) # dopasowanie dat do przewidywań, pomijając pierwsze SEQ_LEN-1 wierszy, w przypadku SEQ_LEN = 20 pomijamy pierwsze 19 wierszy, reset_index aby indeksy pasowały do preds:dates[i] = preds[i]

    pred_df = pd.DataFrame({"Date": dates, "pred_return": preds})
    save_predictions(pred_df, ticker)


def main():

    for ticker in TICKERS:
        train_for_ticker(ticker)


if __name__ == "__main__":
    main()