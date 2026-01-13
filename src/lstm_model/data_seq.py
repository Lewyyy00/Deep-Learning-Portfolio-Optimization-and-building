import pandas as pd
import tensorflow as tf

from src.config.project_variables import FEATURE_COLS, TARGET_COL, SEQ_LEN, BATCH_SIZE

def load_feature_data(ticker="AAPL"):
    df = pd.read_csv(f"data/features/features_{ticker}.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    return df

df = load_feature_data(ticker="AAPL")
train_df = df[df["Date"] <= "2024-12-31"].dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
test_df  = df[df["Date"] >= "2025-01-01"].dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

X_train = train_df[FEATURE_COLS].values
y_train = train_df[TARGET_COL].values

X_test = test_df[FEATURE_COLS].values
y_test = test_df[TARGET_COL].values

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    data=X_train,
    targets=y_train,
    sequence_length=SEQ_LEN,
    sequence_stride=1,
    shuffle=True,
    batch_size=BATCH_SIZE,
)

test_ds = tf.keras.utils.timeseries_dataset_from_array(
    data=X_test,
    targets=y_test,
    sequence_length=SEQ_LEN,
    sequence_stride=1,
    shuffle=False,
    batch_size=BATCH_SIZE,
)

for x_batch, y_batch in train_ds.take(1):
    print("X batch shape:", x_batch.shape)  # (batch, seq_len, n_features)
    print("y batch shape:", y_batch.shape)  # (batch,)