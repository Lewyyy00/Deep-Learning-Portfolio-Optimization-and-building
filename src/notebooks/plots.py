import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/prices.csv")
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