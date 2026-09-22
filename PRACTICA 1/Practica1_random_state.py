import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Carga el dataset
data = fetch_california_housing(as_frame=True)

X = data.data
y = data.target

# Utiliza MedInc como variable de entrada
X_simple = X[["MedInc"]]

# Prueba diferentes valores de random_state
for valor in [10, 20, 42, 100]:

    # Divide los datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_simple,
        y,
        test_size=0.2,
        random_state=valor
    )

    # Crea y entrena el modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Realiza las predicciones
    y_pred = modelo.predict(X_test)

    # Calcula las métricas
    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    # Muestra los resultados
    print("\nRandom State:", valor)
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R²  :", r2)