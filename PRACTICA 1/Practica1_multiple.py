import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Carga el dataset
data = fetch_california_housing(as_frame=True)

X = data.data
y = data.target

# Utiliza todas las variables disponibles
X_multiple = X

# Divide las características y la variable objetivo
# en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_multiple,
    y,
    test_size=0.2,
    random_state=42
)

# Crea un nuevo modelo de regresión lineal
modelo_multiple = LinearRegression()

# Entrena el modelo utilizando todas las variables
modelo_multiple.fit(X_train, y_train)

# Genera predicciones utilizando el conjunto de prueba
y_pred_multiple = modelo_multiple.predict(X_test)

# Calcula el MAE
mae_multiple = mean_absolute_error(
    y_test,
    y_pred_multiple
)

# Calcula el RMSE
rmse_multiple = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_multiple
    )
)

# Calcula el R²
r2_multiple = r2_score(
    y_test,
    y_pred_multiple
)

# Muestra los resultados
print("Resultados del modelo de regresión múltiple")
print("MAE:", mae_multiple)
print("RMSE:", rmse_multiple)
print("R²:", r2_multiple)
print(modelo_multiple.coef_)