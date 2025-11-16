from data_fetcher import *

def plot_stock_prices(data,title):

    data = pd.read_csv(data, index_col=0, parse_dates=True)
   
    plt.figure(figsize=(12, 6))

    for column in data.columns:
        plt.plot(data.index, data[column], label=column)

    plt.title(title)
    plt.xlabel("Data")
    plt.ylabel("Cena")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_stock_prices('log_returns.csv', 'Logarytmiczne Stopy Zwrotu Akcji')