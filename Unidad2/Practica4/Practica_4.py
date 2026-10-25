#--------Paso1--------

# Importar NumPy para generar datos y realizar operaciones matemáticas. 
import numpy as np

# Importar Pandas para organizar los datos en tablas. 
import pandas as pd 

# Importar Matplotlib para crear gráficas. 
import matplotlib.pyplot as plt 

# Importar la función para dividir los datos en entrenamiento y prueba. 
from sklearn.model_selection import train_test_split 

# Importar PolynomialFeatures para generar potencias de la variable. 
from sklearn.preprocessing import PolynomialFeatures

# Importar StandardScaler para estandarizar la variable de entrada. 
from sklearn.preprocessing import StandardScaler 

# Importar Pipeline para encadenar las etapas del modelo. 
from sklearn.pipeline import Pipeline 

# Importar el algoritmo de regresión lineal. 
from sklearn.linear_model import LinearRegression

# Importar las métricas para evaluar las predicciones. 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#--------Paso2--------

# Fijar una semilla para reproducir los mismos datos en cada ejecución. 
np.random.seed(42) 

# Generar 250 temperaturas distribuidas entre 0 y 50 °C. 
temperatura = np.linspace(0, 50, 250)

# Definir una respuesta no lineal ideal del sensor. 
respuesta_ideal = 10 + 1.5 * temperatura + 0.08 * temperatura**2 

# Generar ruido aleatorio para simular variaciones de medición. 
ruido = np.random.normal(loc=0, scale=8, size=250) 

# Sumar el ruido a la respuesta ideal para obtener datos simulados. 
respuesta_sensor = respuesta_ideal + ruido 

# Organizar las temperaturas y las respuestas en una tabla. 
datos = pd.DataFrame({ 
   "temperatura": temperatura, 
   "respuesta_sensor": respuesta_sensor 
}) 
# Mostrar los primeros cinco registros. 
print(datos.head()) 


#--------Paso3--------

# Mostrar la cantidad de filas y columnas del conjunto. 
print("Dimensiones:", datos.shape) 

# Calcular estadísticas descriptivas de las variables. 
print(datos.describe()) 

# Contar los valores faltantes en cada columna. 
print("Valores faltantes:") 
print(datos.isnull().sum()) 


#--------Paso4--------

# Crear una figura de 8 pulgadas de ancho por 5 de alto. 
plt.figure(figsize=(8, 5)) 

# Graficar los datos observados como puntos. 
plt.scatter( 
   datos["temperatura"], 
   datos["respuesta_sensor"], 
   alpha=0.65, 
   label="Mediciones simuladas" 
) 

# Agregar el título de la gráfica. 
plt.title("Respuesta simulada del sensor de temperatura") 

# Etiquetar el eje horizontal. 
plt.xlabel("Temperatura (°C)") 

# Etiquetar el eje vertical. 
plt.ylabel("Respuesta del sensor (unidades arbitrarias)") 

# Mostrar la leyenda de la gráfica. 
plt.legend() 

# Ajustar el diseño para evitar superposiciones. 
plt.tight_layout() 

# Mostrar la gráfica. 
plt.show() 


#--------Paso5--------

# Seleccionar la temperatura como variable de entrada. 
X = datos[["temperatura"]] 

# Seleccionar la respuesta del sensor como variable objetivo. 
y = datos["respuesta_sensor"] 

# Reservar el 20 % de los registros para la evaluación. 
# El 80 % restante se utilizará para entrenar los modelos. 
X_train, X_test, y_train, y_test = train_test_split( 
   X, 
   y, 
   test_size=0.20, 
   random_state=42 
) 

# Mostrar la cantidad de registros de entrenamiento. 
print("Registros de entrenamiento:", len(X_train)) 

# Mostrar la cantidad de registros de prueba. 
print("Registros de prueba:", len(X_test)) 


#--------Paso6--------

# Crear una secuencia de procesamiento para el modelo. 
modelo_grado_2 = Pipeline([ 
   # Estandarizar la temperatura utilizando los datos de entrenamiento. 
   ("escalado", StandardScaler()), 

   # Generar las características de grado 2: x y x². 
   # include_bias=False evita agregar una columna de unos, 
   # porque LinearRegression ya estima el intercepto. 
   ("polinomio", PolynomialFeatures( 
      degree=2, 
      include_bias=False 
    )), 

    # Ajustar una regresión lineal sobre las características generadas. 
    ("regresion", LinearRegression()) 
])

# Entrenar el modelo con las temperaturas y respuestas de entrenamiento. 
modelo_grado_2.fit(X_train, y_train) 

# Generar predicciones para el conjunto de prueba. 
predicciones_grado_2 = modelo_grado_2.predict(X_test) 

# Mostrar las primeras cinco predicciones. 
print("Primeras cinco predicciones:") 
print(predicciones_grado_2[:5])


#--------Paso7--------

# Calcular el error absoluto medio de las predicciones. 
mae_grado_2 = mean_absolute_error(y_test, predicciones_grado_2) 

# Calcular la raíz del error cuadrático medio. 
rmse_grado_2 = np.sqrt(mean_squared_error(y_test, predicciones_grado_2)) 

# Calcular el coeficiente de determinación. 
r2_grado_2 = r2_score(y_test, predicciones_grado_2) 

# Imprimir las métricas con cuatro decimales. 
print(f"MAE: {mae_grado_2:.4f}") 
print(f"RMSE: {rmse_grado_2:.4f}") 
print(f"R²: {r2_grado_2:.4f}")


#--------Paso8--------

# Ordenar las temperaturas de prueba para dibujar una curva continua.
orden = np.argsort(X_test["temperatura"].to_numpy()) 

# Obtener las temperaturas de prueba en orden ascendente. 
temperaturas_ordenadas = X_test["temperatura"].to_numpy()[orden] 

# Reordenar las predicciones de acuerdo con la temperatura. 
predicciones_ordenadas = predicciones_grado_2[orden] 

# Crear la figura de comparación. 
plt.figure(figsize=(8, 5)) 

# Graficar las mediciones reales del conjunto de prueba. 
plt.scatter( 
   X_test["temperatura"], 
   y_test, 
   alpha=0.65, 
   label="Datos de prueba" 
) 

# Graficar la curva estimada por el modelo polinomial. 
plt.plot( 
   temperaturas_ordenadas, 
   predicciones_ordenadas, 
   label="Modelo de grado 2" 
) 

# Agregar título y etiquetas. 
plt.title("Ajuste de regresión polinomial de grado 2") 
plt.xlabel("Temperatura (°C)") 
plt.ylabel("Respuesta del sensor") 

# Mostrar la leyenda. 
plt.legend() 

# Ajustar el diseño. 
plt.tight_layout() 

# Mostrar la gráfica.
plt.show()


#--------Paso9--------

# Definir los grados polinomiales que se desean comparar. 
grados = [1, 2, 3, 5, 10] 
 
# Crear una lista para guardar las métricas de cada modelo. 
resultados = [] 
 
# Crear un diccionario para almacenar los modelos entrenados. 
modelos = {} 
 
# Recorrer cada grado de la lista. 
for grado in grados: 
 
    # Crear el modelo con escalado, características polinomiales 
    # y regresión lineal. 
    modelo = Pipeline([ 
        ("escalado", StandardScaler()), 
        ("polinomio", PolynomialFeatures( 
            degree=grado, 
            include_bias=False 
        )), 
        ("regresion", LinearRegression()) 
    ]) 
 
    # Entrenar el modelo únicamente con los datos de entrenamiento. 
    modelo.fit(X_train, y_train) 
 
    # Calcular el desempeño en el conjunto de entrenamiento. 
    pred_train = modelo.predict(X_train) 
 
    # Calcular el desempeño en el conjunto de prueba. 
    pred_test = modelo.predict(X_test) 
 
    # Calcular el RMSE de entrenamiento. 
    rmse_train = np.sqrt( 
        mean_squared_error(y_train, pred_train) 
    ) 
 
    # Calcular el RMSE de prueba. 
    rmse_test = np.sqrt( 
        mean_squared_error(y_test, pred_test) 
    ) 
 
    # Calcular el R² de prueba. 
    r2_test = r2_score(y_test, pred_test) 
 
    # Guardar las métricas del modelo actual. 
    resultados.append({ 
        "Grado": grado, 
        "RMSE entrenamiento": rmse_train, 
        "RMSE prueba": rmse_test, 
        "R² prueba": r2_test 
    }) 
 
    # Guardar el modelo entrenado para utilizarlo después. 
    modelos[grado] = modelo 
 
# Convertir los resultados en una tabla. 
tabla_resultados = pd.DataFrame(resultados) 

# Mostrar la comparación de los grados polinomiales. 
print(tabla_resultados.round(4)) 


#--------Paso10--------

# Crear una secuencia de temperaturas para dibujar curvas suaves. 
temperaturas_grafica = np.linspace( 
   X["temperatura"].min(), 
   X["temperatura"].max(), 
   300 
) 

# Convertir la secuencia en un DataFrame con el nombre de la variable. 
X_grafica = pd.DataFrame({ 
   "temperatura": temperaturas_grafica 
}) 

# Crear la figura donde se compararán los modelos. 
plt.figure(figsize=(10, 6)) 

# Mostrar todas las mediciones simuladas. 
plt.scatter( 
   X["temperatura"], 
   y, 
   alpha=0.35, 
   label="Datos simulados" 
) 

# Graficar cada modelo con su grado correspondiente. 
for grado in grados: 
   # Generar predicciones sobre la secuencia ordenada. 
   curva = modelos[grado].predict(X_grafica) 
   # Dibujar la curva estimada. 
   plt.plot( 
      temperaturas_grafica, 
      curva, 
      label=f"Grado {grado}" 
) 
   
# Agregar el título y las etiquetas de los ejes. 
plt.title("Comparación de modelos de regresión polinomial") 
plt.xlabel("Temperatura (°C)") 
plt.ylabel("Respuesta del sensor") 

# Mostrar la leyenda de los modelos. 
plt.legend() 

# Ajustar el espacio de la gráfica. 
plt.tight_layout() 

# Mostrar la figura final. 
plt.show()