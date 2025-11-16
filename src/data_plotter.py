from src.data_fetcher import *

def plot_stock_prices(adj_close):
   
    plt.figure(figsize=(12, 6))

    for column in adj_close.columns:
        plt.plot(adj_close.index, adj_close[column], label=column)

    plt.title("Ceny akcji w czasie (Adj Close)")
    plt.xlabel("Data")
    plt.ylabel("Cena")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


tickers = ["AAPL", "MSFT", "GOOGL"]
