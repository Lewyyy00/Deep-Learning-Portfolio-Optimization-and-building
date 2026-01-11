import pandas as pd
import matplotlib.pyplot as plt
import io


# Jeśli masz plik CSV, użyj: df = pd.read_csv('nazwa_pliku.csv')
df = pd.read_csv("data/processed/prices.csv")

# 2. Konwersja kolumny 'Date' na format daty (kluczowe dla poprawnej osi X)
df['Date'] = pd.to_datetime(df['Date'])

# 3. Ustawienie daty jako indeksu (ułatwia rysowanie wielu linii)
df.set_index('Date', inplace=True)

# 4. Tworzenie wykresu
plt.figure(figsize=(12, 6))

# Rysujemy wszystkie kolumny (akcje)
for column in df.columns:
    plt.plot(df.index, df[column], label=column)

# Dodatki do wykresu
plt.title('Cena akcji w czasie', fontsize=14)
plt.xlabel('Data')
plt.ylabel('Cena (USD)')
plt.legend(title='Firmy')
plt.grid(True, linestyle='--', alpha=0.7)

# Wyświetlenie
plt.tight_layout()
plt.show()