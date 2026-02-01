# Deep-Learning-Portfolio-Optimization-and-building

## Opis projektu
Projekt stanowi implementację eksperymentu badawczego poświęconego porównaniu dwóch podejść do konstrukcji i optymalizacji portfela inwestycyjnego:

- klasycznego modelu Markowitza,
- podejścia opartego na prognozach oczekiwanych stóp zwrotu generowanych przez rekurencyjne sieci neuronowe typu LSTM.


Celem projektu była ocena, czy wykorzystanie modeli sekwencyjnych uczenia maszynowego (LSTM) do estymacji parametrów portfela pozwala osiągnąć porównywalne lub lepsze wyniki inwestycyjne względem tradycyjnych metod statystycznych, przy zachowaniu tych samych założeń symulacyjnych.

W pratyce oba podejścia rożróżnia tylko i wyłącznie inna metoda estymacji wektora oczekiwanych stóp zwrotu (kluczowy parametry w optymalizacji portfela) na dzień ustanowiony jako dzień rabalnosowania portfela, co ilustuje poniższy rysunek. Sam mechanizm optymalizacyjny dla obu podejsć jest ten sam i stanowi go algorytm polegajacy na maksymalizacji współczynnika Sharpe'a przy zadanych ograniczeniach portfelowych.

Projekt został przygotowany jako część pracy magisterskiej i umożliwia przeprowadzenie eksperymentów dla dowolnych społek akcyjnych.

## Kluczowe elementy 
### `project_variables.py`

Plik `project_variables.py` pełni rolę centralnego pliku konfiguracyjnego projektu. Zawiera on wszystkie kluczowe parametry eksperymentu, w tym m.in.:

- listę instrumentów finansowych, które ma zawierać portfel (`TICKERS`),
- zakres danych historycznych (`START_DATE`, `END_DATE`),
- parametry rebalansowania portfela (`REBALANCE_STEP`),
- długość okna estymacji kowariancji (`ESTIMATION_WINDOW`),
- hiperparametry modeli LSTM (np. `SEQ_LEN`, `EPOCHS`, `BATCH_SIZE`, i ich siatki),
- ścieżki zapisu danych i wyników.

Zmiana dowolnego parametru w tym pliku umożliwia ponowne uruchomienie całego eksperymentu w nowej konfiguracji, bez konieczności modyfikowania pozostałych części kodu.

---

### `pipeline.py`

Plik `pipeline.py` realizuje kompletny pipeline eksperymentalny. Odpowiada on za sekwencyjne uruchomienie wszystkich etapów badania, w tym:

1. pobieranie danych rynkowych z serwisu Yahoo Finance,
2. preprocessing danych (ceny, logarytmiczne stopy zwrotu),
3. inżynierię cech dla modelu LSTM,
4. tuning hiperparametrów modeli LSTM,
5. trening modeli LSTM i generowanie prognoz,
6. estymację parametrów modelu Markowitza,
7. optymalizację portfela (maksymalizacja Sharpe’a),
8. procedurę backtestu i obliczenie metryk efektywności.

Dzięki temu cały eksperyment może zostać uruchomiony jednym poleceniem.
python -m src.pipeline


## Dzaiałanie i instalacja 
Aby program poprawnie działał zainstaluj go zgodnie z poniższymi krokami.

1. Skopiuj repozytorium z githab-a:

    ```bash
   git clone https://github.com/Lewyyy00/Deep-Learning-Portfolio-Optimization-and-building.git
    ```

2. Następnie należy utworzyć i aktywować środowisko wirtualne:

    ```bash
    # Tworzenie środowiska
    python -m venv venv

    # Aktywacja - Linux / macOS
    source venv/bin/activate

    # Aktywacja - Windows
    venv\Scripts\activate
    ```

3. Zainstaluj wymagane bibloteki
    ```bash
    pip install -r requirements.txt
    ```
4. (opcjonalnie) Zmień hiperparametry modeli i eksperymentu, które znajdują się:

    ```bash
    src/config/project_variables.py
    ```

5. Wejdź do katalogu src i włącz piepline przy pomocy ponizszej komendy
    ```bash
    python -m src.pipeline
    ```