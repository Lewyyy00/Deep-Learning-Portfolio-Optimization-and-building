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