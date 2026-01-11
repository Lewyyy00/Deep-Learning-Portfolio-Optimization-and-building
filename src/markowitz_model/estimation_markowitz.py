
    

class markowitz_optimizer:

    """ 
    Klasa do optymalizacji portfela metodą Markowitza. Niejako silnik aplikacji, który przyjmuje dwa różne obiekty:
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


        
    




