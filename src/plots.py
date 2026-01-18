import pandas as pd
import matplotlib.pyplot as plt

"""df = pd.read_csv("data/processed/prices.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

plt.figure(figsize=(12, 6))
for column in df.columns:
    plt.plot(df.index, df[column], label=column)

plt.title('Cena akcji w czasie', fontsize=14)
plt.xlabel('Data')
plt.ylabel('Cena (USD)')
plt.legend(title='Firmy')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

df = pd.read_csv("data/processed/log_returns.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

plt.figure(figsize=(12, 6))
for column in range(1):
    plt.plot(df.index, df.iloc[:, column], label=df.columns[column])

plt.title(f'Zwroty akcji {df.columns[0]} w czasie', fontsize=14)
plt.xlabel('Data')
plt.ylabel('Zwrot (USD)')
plt.legend(title='Firmy')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

df = pd.read_csv("data/features/features_AAPL.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

plt.figure(figsize=(12, 6))
for column in df.columns[:-1]:
    plt.plot(df.index, df[column], label=column)


plt.title(f'Cechy wejściowe', fontsize=14)
plt.xlabel('Data')
plt.ylabel('Wartość cechy')
plt.legend(title='Cechy wejściowe')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
"""


bt_m = pd.read_csv("data/markowitz/backtests/equity_markowitz.csv", parse_dates=["Date"])
bt_l = pd.read_csv("data/markowitz/backtests/equity_lstm.csv", parse_dates=["Date"])

plt.figure(figsize=(10, 6))
plt.plot(bt_m["Date"], bt_m["PortfolioValue"], label="Portfel Markowitza")
plt.plot(bt_l["Date"], bt_l["PortfolioValue"], label="Portfel LSTM")
plt.xlabel("Data")
plt.ylabel("Wartość portfela")
plt.title("Krzywe wartości portfela – Markowitz vs LSTM")
plt.legend()
plt.grid(True)
plt.show()


bt_m["cum_return"] = bt_m["LogReturn"] / bt_m["LogReturn"].iloc[0] - 1
bt_l["cum_return"] = bt_l["LogReturn"] / bt_l["LogReturn"].iloc[0] - 1

plt.figure(figsize=(10, 6))
plt.plot(bt_m["Date"], bt_m["cum_return"], label="Markowitz")
plt.plot(bt_l["Date"], bt_l["cum_return"], label="LSTM")
plt.xlabel("Data")
plt.ylabel("Skumulowana stopa zwrotu")
plt.title("Skumulowana stopa zwrotu portfeli")
plt.legend()
plt.grid(True)
plt.show()