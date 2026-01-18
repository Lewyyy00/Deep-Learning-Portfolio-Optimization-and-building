import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.lstm_model.lstm_training import load_data, make_dataset, build_model

from src.config.project_variables import (
    FEATURE_COLS, TARGET_COL, BATCH_SIZE, EPOCHS,
    LSTM_FEATURES_SAVE_DIR, TICKERS,
    TEST_START_DATE, TRAINING_END_DATE, VALIDATION_START_DATE, VALIDATION_END_DATE, SEQ_GRID, DROPOUT_GRID, LSTM_UNITS_GRID,TUNING_SAVE_DIR
)

def split_train_test(df,
                     train_end=TRAINING_END_DATE,
                     val_start=VALIDATION_START_DATE,
                     val_end=VALIDATION_END_DATE,
                     test_start=TEST_START_DATE):
    """
    Dzieli dane na zbiory treningowy, walidacyjny i testowy według dat.
    Usuwa wiersze z brakującymi wartościami w FEATURE_COLS i TARGET_COL.
    """
    train_df = df[df["Date"] <= train_end].dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    val_df = df[(df["Date"] >= val_start) & (df["Date"] <= val_end)].dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    test_df = df[df["Date"] >= test_start].dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    return train_df, val_df, test_df


def directional_accuracy(y_true, y_pred):
    # trafność znaku (czy zwrot dodatni/ujemny)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def evaluate_on_val(model, val_df, scaler, seq_len):
    X_val = scaler.transform(val_df[FEATURE_COLS].values)
    y_val = val_df[TARGET_COL].values

    val_ds = make_dataset(X_val, y_val, seq_len=seq_len, shuffle=False)

    preds = model.predict(val_ds, verbose=0).flatten()
    y_true = y_val[seq_len - 1:]  # dopasowanie do końców sekwencji

    mse = float(np.mean((y_true - preds) ** 2))
    mae = float(np.mean(np.abs(y_true - preds)))
    da = directional_accuracy(y_true, preds)

    return mse, mae, da


def tune_for_ticker(ticker: str) -> pd.DataFrame:
    df = load_data(ticker)
    train_df, val_df, _ = split_train_test(df)

    # Skalowanie: fit tylko na train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS].values)
    y_train = train_df[TARGET_COL].values

    results = []

    for lstm_units in LSTM_UNITS_GRID:
        for seq_len in SEQ_GRID:
            train_ds = make_dataset(X_train, y_train, seq_len=seq_len, shuffle=True)

            for dropout in DROPOUT_GRID:
                model = build_model(
                    seq_len=seq_len,
                    n_features=len(FEATURE_COLS),
                    lstm_units=lstm_units,
                    dropout=dropout
                )

                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="loss", patience=2, restore_best_weights=True
                    )
                ]

                model.fit(train_ds, epochs=EPOCHS, verbose=0, callbacks=callbacks)

                mse, mae, da = evaluate_on_val(model, val_df, scaler, seq_len)

                results.append({
                    "ticker": ticker,
                    "seq_len": seq_len,
                    "dropout": dropout,
                    "lstm_units": lstm_units,
                    "val_mse": mse,
                    "val_mae": mae,
                    "val_dir_acc": da
                })

                print(
                    f"[{ticker}] units={lstm_units}, seq={seq_len}, dropout={dropout} "
                    f"-> MSE={mse:.6f}, DA={da:.3f}"
                )

    return pd.DataFrame(results).sort_values(["val_mse", "val_mae"], ascending=True).reset_index(drop=True)


def tuning_main():
    all_rows = []
    for t in TICKERS:
        all_rows.append(tune_for_ticker(t))

    all_df = pd.concat(all_rows, ignore_index=True)

    best_df = (
        all_df.sort_values(["ticker", "val_mse", "val_mae"], ascending=True)
        .groupby("ticker")
        .head(1)
        .reset_index(drop=True)
    )

    path = TUNING_SAVE_DIR
    path.mkdir(parents=True, exist_ok=True)

    out_all = path / "lstm_tuning_results.csv"
    out_best = path / "lstm_tuning_best_per_ticker.csv"
    all_df.to_csv(out_all, index=False)
    best_df.to_csv(out_best, index=False)

    print(f"Zapisano: {out_all}")
    print(f"Zapisano: {out_best}")
    print(best_df)


if __name__ == "__main__":
    tuning_main()

