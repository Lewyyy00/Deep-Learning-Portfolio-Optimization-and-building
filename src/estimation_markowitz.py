from data_fetcher import *
from scipy.optimize import minimize
class indicators:

    def __init__(self,end_date="2024-12-30", window_mean=60, window_sigma=252):
      
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
            end_loc = self.log_returns.index.get_loc(end_date)
        except KeyError:
            raise ValueError(f"Data {end_date} nie występuje w danych.")
        
        # sprawdzamy czy mamy wystarczająco dużo danych wstecz
        if end_loc < max(self.window_mean, self.window_sigma):
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
            end_loc = self.log_returns.index.get_loc(end_date)
        except KeyError:
            raise ValueError(f"Data {end_date} nie występuje w danych.")
        
        # sprawdzamy czy mamy wystarczająco dużo danych wstecz
        if end_loc < max(self.window_mean, self.window_sigma):
            raise ValueError(f"Za mało danych przed {end_date} aby użyć okna.")
        
        sigma_window_vector = self.log_returns.iloc[end_loc - self.window_sigma + 1 : end_loc + 1]
        sigma_window_vector_values = sigma_window_vector.cov().values  # pandas.DataFrame
        return sigma_window_vector_values
    

if __name__ == "__main__":


    indicator_calculator = indicators()
    mean_vector = indicator_calculator.estimate_mean_vector()
    print("Mean Vector:")   
    print(mean_vector)
    cov_matrix = indicator_calculator.compute_cov_vector()
    print("Covariance Matrix:") 
    print(cov_matrix)

