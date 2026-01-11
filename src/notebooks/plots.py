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