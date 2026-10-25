# ============================================================
# RETO PRACTICA 3 - MODELO REDUCIDO DE REGRESION
# ============================================================

#------------PASO 1------------
# Importar NumPy para realizar operaciones numéricas.
import numpy as np

# Importar Pandas para organizar los datos.
import pandas as pd

# Importar el conjunto de datos de California.
from sklearn.datasets import fetch_california_housing

# Importar la función para dividir los datos.
from sklearn.model_selection import train_test_split

# Importar el algoritmo de regresión lineal.
from sklearn.linear_model import LinearRegression

# Importar las métricas de evaluación.
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


#------------PASO 2------------
# Cargar el conjunto de datos.
datos = fetch_california_housing()

# Crear el DataFrame con las variables.
X = pd.DataFrame(
    datos.data,
    columns=datos.feature_names
)

# Crear la variable objetivo.
y = pd.Series(
    datos.target,
    name="MedHouseVal"
)


#------------PASO 3------------
# Dividir los datos usando la misma partición
# que se utilizó en la práctica original.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# MODELO 1: REGRESION SIMPLE
# ============================================================

# Utilizar únicamente MedInc.
X_simple = X[["MedInc"]]

# Dividir los datos.
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple,
    y,
    test_size=0.20,
    random_state=42
)

# Crear el modelo.
modelo_simple = LinearRegression()

# Entrenar el modelo.
modelo_simple.fit(X_train_s, y_train_s)

# Realizar predicciones.
pred_simple = modelo_simple.predict(X_test_s)

# Calcular métricas.
mae_simple = mean_absolute_error(y_test_s, pred_simple)

rmse_simple = np.sqrt(
    mean_squared_error(y_test_s, pred_simple)
)

r2_simple = r2_score(y_test_s, pred_simple)


# ============================================================
# MODELO 2: REGRESION MULTIPLE
# ============================================================

# Crear el modelo.
modelo_multiple = LinearRegression()

# Entrenar utilizando las 8 variables.
modelo_multiple.fit(X_train, y_train)

# Realizar predicciones.
pred_multiple = modelo_multiple.predict(X_test)

# Calcular métricas.
mae_multiple = mean_absolute_error(
    y_test,
    pred_multiple
)

rmse_multiple = np.sqrt(
    mean_squared_error(y_test, pred_multiple)
)

r2_multiple = r2_score(
    y_test,
    pred_multiple
)


# ============================================================
# MODELO 3: MODELO REDUCIDO DEL RETO
# ============================================================

# Definir las cuatro variables seleccionadas.
variables_reto = [
    "MedInc",
    "HouseAge",
    "Latitude",
    "Longitude"
]

# Crear el conjunto de datos reducido.
X_reto = X[variables_reto]

# Dividir los datos utilizando la misma semilla.
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reto,
    y,
    test_size=0.20,
    random_state=42
)

# Crear el modelo reducido.
modelo_reto = LinearRegression()

# Entrenar el modelo.
modelo_reto.fit(X_train_r, y_train_r)

# Realizar predicciones.
pred_reto = modelo_reto.predict(X_test_r)

# Calcular MAE.
mae_reto = mean_absolute_error(
    y_test_r,
    pred_reto
)

# Calcular RMSE.
rmse_reto = np.sqrt(
    mean_squared_error(
        y_test_r,
        pred_reto
    )
)

# Calcular R².
r2_reto = r2_score(
    y_test_r,
    pred_reto
)


# ============================================================
# RESULTADOS
# ============================================================

print("\n========================================")
print("RESULTADOS DE LOS TRES MODELOS")
print("========================================")

print("\nRegresión simple:")
print(f"MAE:  {mae_simple:.4f}")
print(f"RMSE: {rmse_simple:.4f}")
print(f"R²:   {r2_simple:.4f}")

print("\nRegresión múltiple:")
print(f"MAE:  {mae_multiple:.4f}")
print(f"RMSE: {rmse_multiple:.4f}")
print(f"R²:   {r2_multiple:.4f}")

print("\nModelo reducido:")
print(f"MAE:  {mae_reto:.4f}")
print(f"RMSE: {rmse_reto:.4f}")
print(f"R²:   {r2_reto:.4f}")


# ============================================================
# TABLA COMPARATIVA
# ============================================================

comparacion = pd.DataFrame({
    "Modelo": [
        "Regresión simple",
        "Regresión múltiple",
        "Modelo reducido"
    ],

    "Variables": [
        1,
        8,
        4
    ],

    "MAE": [
        mae_simple,
        mae_multiple,
        mae_reto
    ],

    "RMSE": [
        rmse_simple,
        rmse_multiple,
        rmse_reto
    ],

    "R²": [
        r2_simple,
        r2_multiple,
        r2_reto
    ]
})


print("\n========================================")
print("TABLA COMPARATIVA")
print("========================================")

print(
    comparacion.to_string(index=False)
)


# ============================================================
# ANALISIS AUTOMATICO
# ============================================================

print("\n========================================")
print("ANALISIS")
print("========================================")

print(
    "\nEl modelo simple utiliza solamente una variable: MedInc."
)

print(
    "El modelo múltiple utiliza las ocho variables disponibles."
)

print(
    "El modelo reducido utiliza cuatro variables: "
    "MedInc, HouseAge, Latitude y Longitude."
)

print(
    "\nPara MAE y RMSE, un valor menor representa un mejor desempeño."
)

print(
    "Para R², un valor mayor representa un mejor desempeño."
)

print(
    "\nEl modelo reducido permite disminuir la cantidad de variables "
    "manteniendo información importante para realizar las estimaciones."
)
