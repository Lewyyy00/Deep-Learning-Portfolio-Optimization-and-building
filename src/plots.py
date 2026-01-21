import pandas as pd
import matplotlib.pyplot as plt
from src.config.project_variables import BACKTEST_SAVE_DIR

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


bt_m = pd.read_csv(BACKTEST_SAVE_DIR/"equity_markowitz_W252_step5.csv", parse_dates=["Date"])
bt_l = pd.read_csv(BACKTEST_SAVE_DIR/"equity_lstm_seq30_e20_b32_step5.csv", parse_dates=["Date"])

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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Parametry eksperymentu ---
np.random.seed(42)
n_days = 252  # 1 rok sesji
tickers = ["A (wysoka corr)", "B (wysoka corr)", "C (niska corr)", "D (niska corr)"]

# Dzienne oczekiwane stopy zwrotu (przykładowe) i zmienności
mu_daily = np.array([0.0004, 0.00038, 0.00035, 0.00042])   # ~ 8-11% rocznie w przybliżeniu
sig_daily = np.array([0.020, 0.019, 0.018, 0.021])        # dzienna zmienność ~ 1.8%-2.1%

# Macierz korelacji: (A,B) wysoka, (C,D) niska, reszta umiarkowanie niska
corr = np.array([
    [1.00, 0.90, 0.20, 0.15],
    [0.90, 1.00, 0.18, 0.12],
    [0.20, 0.18, 1.00, 0.10],
    [0.15, 0.12, 0.10, 1.00]
])

# Budowa macierzy kowariancji: Sigma = D * Corr * D
D = np.diag(sig_daily)
cov = D @ corr @ D

# Generowanie dziennych zwrotów (wielowymiarowy rozkład normalny)
R = np.random.multivariate_normal(mean=mu_daily, cov=cov, size=n_days)
returns = pd.DataFrame(R, columns=tickers)

# --- Metryki ryzyko-dochód (annualizacja) ---
mean_ann = returns.mean() * 252
vol_ann = returns.std(ddof=1) * np.sqrt(252)

# Korelacje par, które chcemy pokazać
rho_high = returns[tickers[0]].corr(returns[tickers[1]])
rho_low = returns[tickers[2]].corr(returns[tickers[3]])

# --- Wykres ryzyko–dochód ---
fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(vol_ann.values, mean_ann.values)

for t in tickers:
    ax.annotate(t, (vol_ann[t], mean_ann[t]), xytext=(8, 6), textcoords="offset points")

# Linie łączące pary + opis korelacji
x1, y1 = vol_ann[tickers[0]], mean_ann[tickers[0]]
x2, y2 = vol_ann[tickers[1]], mean_ann[tickers[1]]
ax.plot([x1, x2], [y1, y2], linewidth=1.5)
ax.text((x1+x2)/2, (y1+y2)/2, f"ρ≈{rho_high:.2f}", va="bottom", ha="left")

x3, y3 = vol_ann[tickers[2]], mean_ann[tickers[2]]
x4, y4 = vol_ann[tickers[3]], mean_ann[tickers[3]]
ax.plot([x3, x4], [y3, y4], linewidth=1.5)
ax.text((x3+x4)/2, (y3+y4)/2, f"ρ≈{rho_low:.2f}", va="bottom", ha="left")

ax.set_xlabel("Ryzyko (zmienność roczna, σ)")
ax.set_ylabel("Dochód (średnia stopa zwrotu roczna, E[R])")
ax.set_title("Przykład: spółki o wysokiej i niskiej korelacji na wykresie ryzyko–dochód")
ax.grid(True)

plt.tight_layout()
plt.show()
