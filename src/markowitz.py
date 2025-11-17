from data_fetcher import *
import cvxpy as cp



processor = indicators()
log_returns = processor.compute_log_returns()

mu_hat = processor.compute_mu_vector(use_log_returns=True)  # pandas.Series
Sigma_hat = log_returns.cov()    

mu = mu_hat.values          # wektor średnich (n,)
Sigma = Sigma_hat.values    # macierz kowariancji (n, n)
n = len(mu)
asset_labels = list(mu_hat.index)


def compute_efficient_frontier(mu, Sigma, num_points=50, short_selling=False):
    """
    Oblicza punkty granicy efektywnej dla zadanego wektora średnich stóp zwrotu
    i macierzy kowariancji.

    Parametry:
    ----------
    mu : np.ndarray, shape (n,)
        Wektor oczekiwanych stóp zwrotu.
    Sigma : np.ndarray, shape (n, n)
        Macierz kowariancji stóp zwrotu.
    num_points : int
        Liczba punktów na granicy efektywnej.
    short_selling : bool
        Czy dopuszczać krótką sprzedaż (wagi ujemne).

    Zwraca:
    -------
    frontier_risks : list[float]
        Lista odchyleń standardowych portfeli (σ_p).
    frontier_returns : list[float]
        Lista oczekiwanych stóp zwrotu portfeli (μ_p).
    frontier_weights : list[np.ndarray]
        Lista wektorów wag dla kolejnych portfeli na granicy.
    """

    n = len(mu)
    mu_min = mu.min()
    mu_max = mu.max()

    target_returns = np.linspace(mu_min, mu_max, num_points)

    frontier_risks = []
    frontier_returns = []
    frontier_weights = []

    # zmienna decyzyjna: wektory wag
    w = cp.Variable(n)

    for mu_target in target_returns:
        constraints = [
            cp.sum(w) == 1,
            w @ mu == mu_target
        ]
        if not short_selling:
            constraints.append(w >= 0)

        portfolio_variance = cp.quad_form(w, Sigma)
        problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
        problem.solve()

        if w.value is None:
            # problem mógł być niewykonalny dla danego mu_target
            continue

        w_opt = w.value
        mu_p = float(w_opt @ mu)
        sigma_p = float(np.sqrt(w_opt.T @ Sigma @ w_opt))

        frontier_risks.append(sigma_p)
        frontier_returns.append(mu_p)
        frontier_weights.append(w_opt)

    return frontier_risks, frontier_returns, frontier_weights

def plot_efficient_frontier(frontier_risks, frontier_returns, mu, Sigma, asset_labels=None):
    """
    Rysuje granicę efektywną oraz (opcjonalnie) punkty odpowiadające pojedynczym aktywom.

    mu : np.ndarray (n,)
        Wektor średnich stóp zwrotu.
    Sigma : np.ndarray (n, n)
        Macierz kowariancji.
    asset_labels : list[str] lub None
        Nazwy aktywów (np. tickery) do opisania punktów.
    """

    plt.figure(figsize=(10, 6))

    # granica efektywna
    plt.plot(frontier_risks, frontier_returns, linestyle='-', marker='', label="Granica efektywna")

    # pojedyncze aktywa
    asset_risks = np.sqrt(np.diag(Sigma))
    asset_returns = mu

    if asset_labels is None:
        asset_labels = [f"Asset {i}" for i in range(len(mu))]

    plt.scatter(asset_risks, asset_returns, marker='o', label="Pojedyncze aktywa")

    for i, label in enumerate(asset_labels):
        plt.annotate(label, (asset_risks[i], asset_returns[i]),
                     xytext=(5, 5), textcoords="offset points")

    plt.xlabel("Ryzyko (odchylenie standardowe σ)")
    plt.ylabel("Oczekiwana stopa zwrotu μ")
    plt.title("Granica efektywna portfela Markowitza")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. Obliczenie średnich i macierzy kowariancji
    processor = indicators()
    log_returns = processor.compute_log_returns()

    mu_hat = processor.compute_mu_vector(use_log_returns=True)  # pandas.Series
    Sigma_hat = log_returns.cov()    

    mu = mu_hat.values          # wektor średnich (n,)
    Sigma = Sigma_hat.values    # macierz kowariancji (n, n)
    n = len(mu)
    asset_labels = list(mu_hat.index)
    # 2. Obliczenie punktów granicy efektywnej
    frontier_risks, frontier_returns, frontier_weights = compute_efficient_frontier(mu, Sigma, num_points=50, short_selling=False)

    # 3. Wizualizacja
    plot_efficient_frontier(frontier_risks, frontier_returns, mu, Sigma, asset_labels=asset_labels)