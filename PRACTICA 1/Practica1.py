import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
# Importa el modelo de regresión lineal. 
from sklearn.linear_model import LinearRegression 

data = fetch_california_housing (as_frame=True)
#print(data.DESCR)

X = data.data
y = data.target

#print(X.head())
#print("y")
#print(y.head())

#print("dimensiones de X: ", X.shape)
#print("numero de valores de y: ", y.shape)

#print(X.columns)

#print(X.describe())
#print(y.describe())

#plt.figure(figsize=(8, 5))
#plt.hist(y, bins=30)
#plt.xlabel('valor medio de la vivienda')
#plt.ylabel('Frecuencia')
#plt.title('Distribución del valor medio de las viviendas')
#plt.show()

#plt.figure(figsize=(8, 5))
#plt.scatter(X['MedInc'], y, alpha=0.3)
#plt.xlabel('Ingreso medio')
#plt.ylabel('Valor medio de la vivienda')
#plt.title('Relación entre ingreso y valor de la vivienda')
#plt.show()

X_simple = X[['MedInc']]

# Importa la función para dividir un conjunto de datos. 
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(
    X_simple,       # Variables de entrada. 
    y,              # Variable objetivo. 
    test_size=0.2,  # Utiliza el 20 % de los datos para prueba. 
    random_state=42 # Permite reproducir exactamente la misma división. 
)

# Crea un objeto que representa el modelo de regresión lineal. 
modelo = LinearRegression()

# Entrena el modelo utilizando las variables de entrada y los valores conocidos. 
modelo.fit(X_train, y_train) 

# Muestra el valor del intercepto β0. 
print("Intercepto:", modelo.intercept_) 

# Muestra el coeficiente β1 asociado con MedInc. 
print("Coeficiente:", modelo.coef_[0])

# Utiliza el modelo entrenado para generar predicciones. 
y_pred = modelo.predict(X_test)

# Muestra los primeros diez valores reales. 
print("Valores reales:") 
print(y_test.iloc[:10].values) 
# Muestra las primeras diez predicciones. 
print("Predicciones:")
print(y_pred[:10])

# Importa métricas utilizadas para evaluar modelos de regresión.
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score) 
# Calcula el error absoluto medio. 
mae = mean_absolute_error(y_test, y_pred) 
# Calcula el error cuadrático medio. 
mse = mean_squared_error(y_test, y_pred)
# Calcula la raíz cuadrada del error cuadrático medio. 
rmse = np.sqrt(mse)
# Calcula el coeficiente de determinación. 
r2 = r2_score(y_test, y_pred)

# Imprime las cuatro métricas calculadas. 
print("MAE :", mae) 
print("MSE :", mse) 
print("RMSE:", rmse)
print("R2  :", r2)

# Crea una figura. 
plt.figure(figsize=(8, 5)) 

# Grafica los valores reales del conjunto de prueba.
plt.scatter( 
   X_test["MedInc"], 
   y_test, 
   alpha=0.4, 
   label="Datos reales" 
) 

# Ordena los valores de X para que la línea se dibuje correctamente. 
orden = np.argsort(X_test["MedInc"].values) 
# Dibuja la línea correspondiente a las predicciones. 
plt.plot( 
   X_test["MedInc"].values[orden], 
   y_pred[orden], 
   linewidth=2, 
   label="Regresión lineal" 
) 

# Etiqueta el eje X. 
plt.xlabel("Ingreso medio") 
# Etiqueta el eje Y. 
plt.ylabel("Valor medio de la vivienda") 
# Agrega el título. 
plt.title("Modelo de regresión lineal") 
# Muestra la leyenda. 
plt.legend() 
# Muestra la gráfica. 
plt.show()

# Crea una tabla con los valores reales y las predicciones. 
comparacion = pd.DataFrame({ 
"Real": y_test.values, 
"Prediccion": y_pred 
}) 
# Calcula el error entre el valor real y el predicho. 
comparacion["Error"] = ( 
comparacion["Real"] - comparacion["Prediccion"] 
) 
# Muestra los primeros diez registros. 
print(comparacion.head(10))

# Calcula el valor absoluto del error. 
comparacion["Error_Absoluto"] = ( 
comparacion["Error"].abs() 
) 
# Ordena los registros de mayor a menor error. 
mayores_errores = comparacion.sort_values( 
"Error_Absoluto", 
ascending=False 
) 
# Muestra los diez casos con mayor error. 
print(mayores_errores.head(10)) 