from data_fetcher import *
from scipy.optimize import minimize
class indicators:

    def __init__(self,end_date="2024-12-30", window_mean=252, window_sigma=252):


        """
        Klasa indicators jest odpowiedzialna za część czysto statystyczną modelu Markowitza. Jej zadaniem: 
            - jest wczytanie wczytanie historycznych danych cenowych aktywów - _load_adj_close(),
            - obliczenie logarytmicznych stóp zwrotu - compute_log_returns(),
            - oszaczowanie:
                wektora średnich stóp zwrotu - estimate_mean_vector(),
                macierzy kowariancji stóp zwrotu - estimate_cov_matrix().
            - estymacja:
                oczekiwanej stopy zwrotu portfela - compute_expected_return(),
                wariancji portfela - compute_portfolio_variance().

        Parametry:
            end_date - data końcowa okresu, z którego mają być estymowane parametry.
            W modelu Markowitza zwykle zakładamy, że w danym dniu inwestor patrzy na pewne 
            okno historyczne „wstecz” (np. ostatnie 252 dni) i na tej podstawie szacuje średnie zwroty i ryzyko.

            Okres historyczny, z którego liczysz średnie i kowariancję, kończy się właśnie w end_date.

            window_mean - długość okna (w dniach) używana do estymacji wektora średnich stóp zwrotu μ.
            Przykład: window_mean = 252 oznacza, że średnia stopa zwrotu dla każdego aktywa liczona jest 
            na podstawie 252 ostatnich dni notowań.

            window_sigma - długość okna (w dniach) używana do estymacji macierzy kowariancji Σ.
            Nie musi być równa window_mean, możesz np. stosować dłuższe okno do kowariancji, a krótsze 
            do średnich.
        
        
        """
      
        self.adj_close_path = "adj_close.csv"
        self.adj_close = self._load_adj_close()
        self.end_date = end_date
        self.window_mean = window_mean
        self.window_sigma = window_sigma


        self.log_returns = None

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

        end_loc = self._handle_missing_data()
        
        # sprawdzamy czy mamy wystarczająco dużo danych wstecz
        if end_loc < self.window_mean:
            raise ValueError(f"Za mało danych przed {end_loc} aby użyć okna.")
        
        mean_window_vector = self.log_returns.iloc[end_loc - self.window_mean + 1 : end_loc + 1]
        mean_window_vector_values = mean_window_vector.mean().values  # numpy vector shape (N,)
        return mean_window_vector_values
    
    def estimate_cov_matrix(self):
        """
        Zwraca macierz kowariancji stóp zwrotu dla każdego aktywa z całego okresu.

                AAPL     GOOGL      MSFT
        AAPL   0.000398  0.000267  0.000288
        GOOGL  0.000267  0.000420  0.000295
        MSFT   0.000288  0.000295  0.000370

        """

        if self.log_returns is None:
            self.compute_log_returns()

        end_loc = self._handle_missing_data()
        
        # sprawdzamy czy mamy wystarczająco dużo danych wstecz
        if end_loc < self.window_sigma:
            raise ValueError(f"Za mało danych przed {end_loc} aby użyć okna.")
        
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
        cov_matrix = self.estimate_cov_matrix()
        return weights.T @ cov_matrix @ weights
    

class markowitz_optimizer:


    """ Klasa do optymalizacji portfela metodą Markowitza. Niejako silnik aplikacji, który przyjmuje dwa różne obiekty:
        -indicators do przeprowadzenia tradycyjnej optymalizacji Markowitza,
        -LSTM do optymalizacji portfela na podstawie prognozowanych stóp zwrotu z modelu LSTM.
        
        Głownym celem jest dobranie wag portfela maksymalizujących Sharpe Ratio
        
        """
    

    def __init__(self, max_single_asset_weight, allow_short_sales = False, lstm = False, risk_free_rate=0.01):

        self.risk_free_rate = risk_free_rate
        self.allow_short_sales = allow_short_sales
        self.max_single_asset_weight = max_single_asset_weight if max_single_asset_weight is not None else 0.4


        if lstm == False: #Markowitz

            self.indicator_calculator = indicators()

            if self.indicator_calculator.log_returns is None:
                self.indicator_calculator.compute_log_returns()
            
            self.asset_names = list(self.indicator_calculator.log_returns.columns) #nazwy aktywów
            self.number_of_assets = len(self.asset_names) #liczba aktywów
            

        else: #LSTM
            self.indicator_calculator = None  
            self.asset_names = None
            self.number_of_assets = None

    def compute_portfolio_sharpe(self, weights):
        """
        Oblicza Sharpe Ratio portfela na podstawie wag, oczekiwanej stopy zwrotu i wariancji.
        Zgodnie ze wzorem: Sharpe = (E[R_p] - R_f) / σ_p
        """
        expected_return = self.indicator_calculator.compute_expected_return(weights)
        variance = self.indicator_calculator.compute_portfolio_variance(weights)
        sharpe_ratio = (expected_return - self.risk_free_rate) / np.sqrt(variance)
        return sharpe_ratio
    
    def compute_negative_sharpe(self, weights):
        """
        Oblicza negatywne Sharpe Ratio portfela (do minimalizacji).
        """
        return -self.compute_portfolio_sharpe(weights)
    
    def optimize_portfolio_with_max_sharpe(self):

        def weights_sum_to_one(weights): # wagi sumują się do 1
            return np.sum(weights) - 1.0

        equality_constraint = {
            "type": "eq",
            "fun": weights_sum_to_one
        }

        #Ograniczenia brzegowe na każdą wagę (bounds)
        if self.allow_short_sales: # Dopuszczamy krótką sprzedaż – wagi mogą być ujemne. Przykładowo: od -1 do +1.
            
            bounds = [(-1.0, 1.0) for _ in range(self.number_of_assets)]
        else: # Portfel long-only: 0 <= w_i <= max_single_asset_weight
            
            bounds = [(0.0, self.max_single_asset_weight) for _ in range(self.number_of_assets)]
            max_possible_sum = self.number_of_assets * self.max_single_asset_weight # Sprawdzamy, czy przy takich bounds suma wag w ogóle może osiągnąć 1.
            if max_possible_sum < 1.0:
                raise ValueError(
                    f"Ograniczenia są niewykonalne: liczba aktywów * max_single_asset_weight "
                    f"= {max_possible_sum:.2f} < 1.0. Zwiększ max_single_asset_weight."
                )
        

        initial_weights = np.repeat(1.0 / self.number_of_assets, self.number_of_assets)

        # minimalizacja negatywnego Sharpe Ratio
        optimization_result = minimize(
            fun=self.compute_negative_sharpe,
            x0=initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=[equality_constraint],
            options={
                "maxiter": 1000,
                "ftol": 1e-9
            }
        )

        if not optimization_result.success:
            raise RuntimeError(f"Optymalizacja się nie powiodła: {optimization_result.message}")
        
        optimal_weights_array = optimization_result.x

        # Zaokrąglenie bardzo małych liczb do zera - wtedy nie sumują się do 1 dokładnie
        optimal_weights_array[np.abs(optimal_weights_array) < 1e-10] = 0.0

        # Zwracamy wynik jako pd.Series z nazwami aktywów
        optimal_weights = pd.Series(
            data=optimal_weights_array,
            index=self.asset_names,
            name="Optimal_Markowitz_Weights"
        )

        return optimal_weights



#Tworzymy obiekt indicators
ind = indicators(
    end_date="2024-12-30",
    window_mean=252,
    window_sigma=252
)

#Tworzymy optymalizator Markowitza i wyznaczamy wagi portfela max Sharpe 
optimizer = markowitz_optimizer()
optimal_weights = optimizer.optimize_portfolio_with_max_sharpe()

print("Optymalne wagi portfela Markowitza (max Sharpe):")
print(optimal_weights)
print("Suma wag:", optimal_weights.sum())















"""indicator_calculator = indicators()
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
"""


        
    




