from data_fetcher import *
from scipy.optimize import minimize
class indicators:

    def __init__(self,end_date="2024-12-30", window_mean=252, window_sigma=252):
      
        self.adj_close_path = "adj_close.csv"
        self.adj_close = self._load_adj_close()
        self.end_date = end_date
        self.window_mean = window_mean
        self.window_sigma = window_sigma


        self.log_returns = None
        self.simple_returns = None

    def _load_adj_close(self):
     
        try:
            df = pd.read_csv(self.adj_close_path, index_col=0)
            df.index = pd.to_datetime(df.index)

        except FileNotFoundError:
            raise FileNotFoundError(f"Nie znaleziono pliku {self.adj_close_path}.")
        
        return df
    
    def _handle_missing_data(self):
        """
        Zwraca indeks (pozycję) ostatniego dostępnego dnia sesyjnego
        <= self.end_date. - jeśli self.end_date nie jest dniem sesyjnym 
        bo np. weekend lub święto.
        """
        if self.log_returns is None:
            self.compute_log_returns()

        idx = self.log_returns.index

        if self.end_date in idx:
            effective_date = self.end_date
        else:
            mask = idx <= self.end_date
            if not mask.any():
                raise ValueError(f"Brak danych przed {self.end_date}.")
            effective_date = idx[mask][-1]

        return idx.get_loc(effective_date)


    def compute_log_returns(self, save_csv=True, csv_path="log_returns.csv"):

        """
        Oblicza logarytmiczne stopy zwrotu: r_t = ln(P_t / P_{t-1}) na podstawie danych Adj Close.

        """
        
        log_ret = np.log(self.adj_close / self.adj_close.shift(1)).dropna()
        self.log_returns = log_ret

        if save_csv:
            log_ret.to_csv(csv_path)

        return log_ret
    
    def estimate_mean_vector(self):
        
        """
        Zwraca wektor średnich stóp zwrotu dla każdego aktywa z całego podanego okna.

        AAPL     0.000983
        GOOGL    0.000812
        MSFT     0.000803

        """

        if self.log_returns is None:
            self.compute_log_returns()

        end_date = pd.to_datetime(self.end_date)

        try:
            end_loc = self._handle_missing_data()
        except KeyError:
            raise ValueError(f"błąd z datą: {end_date}.")
        
        # sprawdzamy czy mamy wystarczająco dużo danych wstecz
        if end_loc < self.window_mean:
            raise ValueError(f"Za mało danych przed {end_date} aby użyć okna.")
        
        mean_window_vector = self.log_returns.iloc[end_loc - self.window_mean + 1 : end_loc + 1]
        mean_window_vector_values = mean_window_vector.mean().values  # numpy vector shape (N,)
        return mean_window_vector_values
    
    def estimate_cov_vector(self):
        """
        Zwraca macierz kowariancji stóp zwrotu dla każdego aktywa z całego okresu.

                AAPL     GOOGL      MSFT
        AAPL   0.000398  0.000267  0.000288
        GOOGL  0.000267  0.000420  0.000295
        MSFT   0.000288  0.000295  0.000370

        """

        if self.log_returns is None:
            self.compute_log_returns()

        end_date = pd.to_datetime(self.end_date)

        try:
            end_loc = self._handle_missing_data()
        except KeyError:
            raise ValueError(f"błąd z datą: {end_date}.")
        
        # sprawdzamy czy mamy wystarczająco dużo danych wstecz
        if end_loc < self.window_sigma:
            raise ValueError(f"Za mało danych przed {end_date} aby użyć okna.")
        
        sigma_window_vector = self.log_returns.iloc[end_loc - self.window_sigma + 1 : end_loc + 1]
        sigma_window_vector_values = sigma_window_vector.cov().values  # pandas.DataFrame
        return sigma_window_vector_values
    
    def compute_expected_return(self, weights):
        """
        Oblicza oczekiwaną stopę zwrotu portfela na podstawie wag i wektora średnich stóp zwrotu.
        Zgodnie ze wzorem: E[R_p] = w^T * μ
        """
        return weights.T @ self.estimate_mean_vector()
    
    def compute_portfolio_variance(self, weights):
        """
        Oblicza wariancję portfela na podstawie wag i macierzy kowariancji stóp zwrotu.
        Zgodnie ze wzorem: σ_p^2 = w^T * Σ * w
        """
        cov_matrix = self.estimate_cov_vector()
        return weights.T @ cov_matrix @ weights
    

indicator_calculator = indicators()
weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1,0.1])

    
def sharpe_ratio(weights, risk_free_rate=0.01):
    exp_ret = indicator_calculator.compute_expected_return(weights)
    var = indicator_calculator.compute_portfolio_variance(weights)
    return (exp_ret - risk_free_rate) / np.sqrt(var)

def neg_sharpe_ratio():
    return -sharpe_ratio()

def neg_sharpe_ratio(weights, indicator_calculator, risk_free_rate=0.01):
    expected_return = indicator_calculator.compute_expected_return(weights)
    variance = indicator_calculator.compute_portfolio_variance(weights)
    sharpe = (expected_return - risk_free_rate) / np.sqrt(variance)
    return -sharpe

constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.4) for _ in range(len(weights))]
initial_weights = np.array([1/len(weights)]*len(weights))

result = minimize(
    neg_sharpe_ratio,
    initial_weights,
    args=(indicator_calculator, 0.01),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x
print("Optymalne wagi maksymalizujące Sharpe'a:", optimal_weights)
expected_returnn = indicator_calculator.compute_expected_return(optimal_weights)
variancen = indicator_calculator.compute_portfolio_variance(optimal_weights)



        
    




