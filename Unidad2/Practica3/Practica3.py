#------------PASO 1------------
# Importar NumPy para realizar operaciones numéricas. 
import numpy as np 

# Importar Pandas para organizar y analizar los datos. 
import pandas as pd 

# Importar Matplotlib para construir gráficas. 
import matplotlib.pyplot as plt

# Importar la función para cargar el conjunto de datos. 
from sklearn.datasets import fetch_california_housing 

# Importar la función para dividir los datos. 
from sklearn.model_selection import train_test_split 

# Importar el algoritmo de regresión lineal. 
from sklearn.linear_model import LinearRegression 

# Importar métricas para evaluar las predicciones. 
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)


#------------PASO 2------------

# Cargar el conjunto de datos desde Scikit-learn. 
# La primera descarga puede requerir conexión a internet. 
datos = fetch_california_housing() 

# Convertir las variables explicativas en un DataFrame. 
X = pd.DataFrame(datos.data, columns=datos.feature_names)

# Crear una serie con la variable objetivo. 
y = pd.Series(datos.target, name="MedHouseVal") 

# Mostrar las primeras cinco filas de las variables de entrada. 
print("Variables explicativas:") 
print(X.head()) 

# Mostrar los primeros cinco valores objetivo. 
print("Variable objetivo:") 
print(y.head()) 


#------------PASO 3------------

# Mostrar el número de filas y columnas de las variables de entrada. 
print("Dimensiones de X:", X.shape) 

# Mostrar la cantidad de valores objetivo. 
print("Cantidad de valores objetivo:", y.shape[0]) 

# Mostrar los nombres de las variables. 
print("Variables disponibles:") 
print(X.columns.tolist()) 

# Obtener estadísticas descriptivas de cada variable. 
print(X.describe()) 

# Contar los valores faltantes en cada columna. 
print("Valores faltantes por variable:") 
print(X.isnull().sum()) 

# Verificar si existen valores faltantes en la variable objetivo. 
print("Valores faltantes en y:", y.isnull().sum())


#------------PASO 4------------

# Crear una figura para la distribución del valor de las viviendas. 
plt.figure(figsize=(8, 5)) 

# Dibujar un histograma con 30 intervalos. 
plt.hist(y, bins=30, edgecolor="black") 

# Agregar el título de la gráfica. 
plt.title("Distribución del valor medio de las viviendas")

# Identificar el eje horizontal. 
plt.xlabel("Valor medio (unidades de 100 000 dólares)") 

# Identificar el eje vertical. 
plt.ylabel("Frecuencia") 

# Ajustar los elementos para evitar superposiciones. 
plt.tight_layout() 

# Mostrar la gráfica. 
plt.show() 


#------------PASO 5------------

# Dividir los datos en entrenamiento y prueba. 
# El 80 % se utiliza para entrenar y el 20 % para evaluar. 
X_train, X_test, y_train, y_test = train_test_split( 
   X, 
   y, 
   test_size=0.20, 
   random_state=42 
) 

# Mostrar el número de registros de entrenamiento. 
print("Registros de entrenamiento:", X_train.shape[0]) 

# Mostrar el número de registros de prueba. 
print("Registros de prueba:", X_test.shape[0]) 


#------------PASO 6------------

# Crear una instancia del modelo de regresión lineal. 
modelo = LinearRegression() 

# Ajustar el modelo utilizando los datos de entrenamiento. 
# El algoritmo aprende los coeficientes de las ocho variables. 
modelo.fit(X_train, y_train) 

# Mostrar el intercepto aprendido por el modelo. 
print("Intercepto:", modelo.intercept_) 

# Mostrar los coeficientes de cada variable explicativa. 
print("Coeficientes:") 

# Asociar cada nombre de variable con su coeficiente. 
for nombre, coeficiente in zip(X.columns, modelo.coef_ ): 

   # Imprimir el nombre y el coeficiente con cuatro decimales. 
   print(f"{nombre}: {coeficiente:.4f}") 


#------------PASO 7------------

# Utilizar el modelo entrenado para predecir los valores de prueba. 
y_pred = modelo.predict(X_test) 

# Crear una tabla con los valores reales y las predicciones. 
resultados = pd.DataFrame({"Valor real": y_test, "Valor predicho": y_pred}) 

# Mostrar las primeras diez comparaciones. 
print(resultados.head(10)) 


#------------PASO 8------------

# Calcular el error absoluto medio. 
mae = mean_absolute_error(y_test, y_pred) 

# Calcular la raíz del error cuadrático medio.  
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 

# Calcular el coeficiente de determinación. 
r2 = r2_score(y_test, y_pred) 

# Mostrar las métricas obtenidas. 
print(f"MAE: {mae:.4f}") 
print(f"RMSE: {rmse:.4f}") 
print(f"R²: {r2:.4f}") 


#------------PASO 9------------

# Crear una figura para comparar los valores reales y predichos. 
plt.figure(figsize=(7, 6)) 

# Graficar los valores reales en el eje horizontal y las predicciones en el eje vertical. 
plt.scatter(y_test, y_pred, alpha=0.35) 

# Calcular los límites comunes de ambos ejes. 
limite_min = min(y_test.min(), y_pred.min()) 
limite_max = max(y_test.max(), y_pred.max()) 

# Dibujar una línea de referencia para predicciones perfectas. 
plt.plot( 
[limite_min, limite_max], 
[limite_min, limite_max], 
linestyle="--" 
) 

# Agregar título y nombres de los ejes. 
plt.title("Valores reales frente a valores predichos") 
plt.xlabel("Valor real") 
plt.ylabel("Valor predicho") 

# Mostrar la gráfica. 
plt.tight_layout() 
plt.show()


#------------PASO 10------------

# Calcular los residuos del conjunto de prueba. 
residuos = y_test - y_pred 

# Crear la gráfica de residuos frente a predicciones. 
plt.figure(figsize=(8, 5)) 

# Mostrar la relación entre predicciones y errores. 
plt.scatter(y_pred, residuos, alpha=0.35) 

# Dibujar la línea horizontal que representa error cero. 
plt.axhline(y=0, linestyle="--") 

# Agregar etiquetas y título. 
plt.title("Análisis de residuos del modelo") 
plt.xlabel("Valor predicho") 
plt.ylabel("Residuo: valor real - valor predicho") 

# Ajustar el diseño y mostrar la gráfica. 
plt.tight_layout() 
plt.show() 


#------------PASO 11------------

# Seleccionar únicamente la variable MedInc para el modelo simple. 
X_simple = X[["MedInc"]] 

# Dividir la variable simple usando la misma proporción y semilla. 
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
   X_simple, 
   y, 
   test_size=0.20, 
   random_state=42 
) 

# Crear el modelo de regresión lineal simple. 
modelo_simple = LinearRegression() 

# Entrenar el modelo con una sola variable. 
modelo_simple.fit(X_train_s, y_train_s) 

# Generar predicciones para el conjunto de prueba. 
pred_simple = modelo_simple.predict(X_test_s) 

# Calcular el MAE del modelo simple. 
mae_simple = mean_absolute_error(y_test_s, pred_simple) 

# Calcular el RMSE del modelo simple. 
rmse_simple = np.sqrt(mean_squared_error(y_test_s, pred_simple))

# Calcular el R² del modelo simple. 
r2_simple = r2_score(y_test_s, pred_simple) 

# Crear una tabla que compare ambos modelos. 
comparacion = pd.DataFrame({ 
"Modelo": ["Regresión simple", "Regresión múltiple"], 
"MAE": [mae_simple, mae], 
"RMSE": [rmse_simple, rmse], 
"R²": [r2_simple, r2] 
}) 

# Mostrar los resultados para su análisis. 
print(comparacion)