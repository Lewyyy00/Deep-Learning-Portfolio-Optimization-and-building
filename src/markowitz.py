from data_fetcher import *
from estimation_markowitz import *
import cvxpy as cp

class MarkowitzPortfolioOptimizer:
    """
    Klasa do optymalizacji portfela metodą Markowitza.
    """

    def __init__(self):
        """
        Inicjalizuje obiekt optymalizatora Markowitza, pobierając dane 
        stóp zwrotu, kowariacje i ich statystyki z klasy indicators.

        ----------
        ExpectedValueVector : np.ndarray, shape (n,)
            Wektor oczekiwanych stóp zwrotu.
        covarianceMatrix : np.ndarray, shape (n, n)
            Macierz kowariancji stóp zwrotu.
        numberOfAssets : int
            Liczba aktywów w portfelu.
        asset_labels : list[str]
            Nazwy aktywów (np. tickery).
        """
        processor = indicators()
        self.log_returns = processor.compute_log_returns()

        #Wektor oczekiwanych stóp zwrotu.
        self.expectedValueVector = processor.compute_mean_vector(use_log_returns=True)
        self.expectedValueVectorValues = self.expectedValueVector.values

        #Macierz kowariancji stóp zwrotu.
        self.covarianceMatrix = processor.compute_cov_vector()
        self.covarianceMatrixValues = processor.compute_cov_vector().values

        self.numberOfAssets = len(self.expectedValueVector)
        self.asset_labels = list(self.expectedValueVector.index)

    def compute_efficient_frontier(self,num_points=50, short_selling=False):

        """
        Oblicza punkty granicy efektywnej dla zadanego wektora średnich stóp zwrotu
        i macierzy kowariancji.

        Parametry:
        ----------
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

        mu_min = self.expectedValueVector.min()
        mu_max = self.expectedValueVector.max()

        target_returns = np.linspace(mu_min, mu_max, num_points) #linspace tworzy wektor z określoną liczbą równomiernie rozmieszczonych punktów pomiędzy wartością początkową a końcową

        frontier_risks = []
        frontier_returns = []
        frontier_weights = []

        w = cp.Variable(self.numberOfAssets) # zmienna decyzyjna: wektory wag
        
        for target_return in target_returns:
            constraints = [ #ograniczenia
                cp.sum(w) == 1,
                w @ self.expectedValueVector == target_return
            ]
            if not short_selling:
                constraints.append(w >= 0)

            portfolio_variance = cp.quad_form(w, self.covarianceMatrixValues)
            problem = cp.Problem(cp.Minimize(portfolio_variance), constraints) #problem optymalizacyjny: minimalizacja wariancji portfela przy zadanych ograniczeniach
            problem.solve()

            if w.value is None:
                # problem mógł być niewykonalny dla danego target_return
                continue

            w_opt = w.value
            mu_p = float(w_opt @ self.expectedValueVector)
            sigma_p = float(np.sqrt(w_opt.T @ self.covarianceMatrixValues @ w_opt))

            frontier_risks.append(sigma_p)
            frontier_returns.append(mu_p)
            frontier_weights.append(w_opt)

        return frontier_risks, frontier_returns, frontier_weights
    
    def generate_random_portfolios(self, n_portfolios=10000, seed=42):
        """
        Generuje losowe portfele dopuszczalne (wagi >=0, suma wag = 1)
        i zwraca ich ryzyko i oczekiwaną stopę zwrotu.

        Parametry
        ---------
        n_portfolios : int
            Liczba losowych portfeli do wygenerowania.
        seed : int
            Ziarno generatora losowego dla powtarzalności wyników.

        Zwraca
        ------
        risks : np.ndarray (n_portfolios,)
            Odchylenia standardowe portfeli.
        returns : np.ndarray (n_portfolios,)
            Oczekiwane stopy zwrotu portfeli.
        weights : np.ndarray (n_portfolios, n)
            Macierz wag portfeli.
        """

        np.random.seed(seed)
        

        # Losowanie wag z rozkładu Dirichleta: w_i >= 0, sum(w)=1
        W = np.random.dirichlet(alpha=np.ones(self.numberOfAssets), size=n_portfolios)

        # Stopy zwrotu portfela: mu_p = w^T mu
        port_returns = W @ self.expectedValueVector

        # Wariancje portfela: sigma^2 = w^T Sigma w
        port_variances = np.einsum('ij,jk,ik->i', W, self.covarianceMatrixValues, W)
        port_risks = np.sqrt(port_variances)

        return port_risks, port_returns, W
    
    def plot_feasible_set_and_efficient_frontier(self,n_portfolios=10000,num_points_frontier=50):
    
        """
        Rysuje:
        - "chmurę" punktów reprezentującą cały zbiór portfeli dopuszczalnych,
        - granicę efektywną jako górną obwiednię,
        - punkty pojedynczych aktywów.

        Parametry
        ----------
        n_portfolios : int
            Liczba losowych portfeli do wygenerowania (Monte Carlo).
        num_points_frontier : int
            Liczba punktów na granicy efektywnej.
        """

        print(n_portfolios)
        asset_labels = self.asset_labels 
        if asset_labels is None:
            asset_labels = [f"Asset {i}" for i in range(self.numberOfAssets)]

        # Zbiór portfeli dopuszczalnych (Monte Carlo)
        feasible_risks, feasible_returns, _ = self.generate_random_portfolios(n_portfolios=n_portfolios)

        # Granica efektywna
        frontier_risks, frontier_returns, _ = self.compute_efficient_frontier(num_points=num_points_frontier, short_selling=False)

        # Pojedyncze aktywa
        asset_risks = np.sqrt(np.diag(self.covarianceMatrixValues))
        asset_returns = self.expectedValueVector

        # Wykres
        plt.figure(figsize=(10, 6))

        # cała chmura portfeli dopuszczalnych
        plt.scatter(feasible_risks, feasible_returns,
                    s=5, alpha=0.3, label="Portfele dopuszczalne (Monte Carlo)")

        # granica efektywna
        plt.plot(frontier_risks, frontier_returns,
                linewidth=2.0, label="Granica efektywna")

        # pojedyncze aktywa
        plt.scatter(asset_risks, asset_returns,
                    marker='o', s=40, label="Pojedyncze aktywa")

        for i, label in enumerate(asset_labels):
            plt.annotate(label,
                        (asset_risks[i], asset_returns[i]),
                        xytext=(5, 5),
                        textcoords="offset points")

        plt.xlabel("Ryzyko (odchylenie standardowe σ)")
        plt.ylabel("Oczekiwana stopa zwrotu μ")
        plt.title("Zbiór portfeli dopuszczalnych i granica efektywna (Markowitz)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()