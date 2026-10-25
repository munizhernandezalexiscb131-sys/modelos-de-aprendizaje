#--------Paso1--------

# Importar NumPy para generar datos y realizar operaciones matemÃ¡ticas. 
import numpy as np

# Importar Pandas para organizar los datos en tablas. 
import pandas as pd 

# Importar Matplotlib para crear grÃ¡ficas. 
import matplotlib.pyplot as plt 

# Importar la funciÃ³n para dividir los datos en entrenamiento y prueba. 
from sklearn.model_selection import train_test_split 

# Importar PolynomialFeatures para generar potencias de la variable. 
from sklearn.preprocessing import PolynomialFeatures

# Importar StandardScaler para estandarizar la variable de entrada. 
from sklearn.preprocessing import StandardScaler 

# Importar Pipeline para encadenar las etapas del modelo. 
from sklearn.pipeline import Pipeline 

# Importar el algoritmo de regresiÃ³n lineal. 
from sklearn.linear_model import LinearRegression

# Importar las mÃ©tricas para evaluar las predicciones. 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#--------Paso2--------

# Fijar una semilla para reproducir los mismos datos en cada ejecuciÃ³n. 
np.random.seed(42) 

# Generar 250 temperaturas distribuidas entre 0 y 50 Â°C. 
temperatura = np.linspace(0, 50, 250)

# Definir una respuesta no lineal ideal del sensor. 
respuesta_ideal = 10 + 1.5 * temperatura + 0.08 * temperatura**2 

# Generar ruido aleatorio para simular variaciones de mediciÃ³n. 
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

# Calcular estadÃ­sticas descriptivas de las variables. 
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

# Agregar el tÃ­tulo de la grÃ¡fica. 
plt.title("Respuesta simulada del sensor de temperatura") 

# Etiquetar el eje horizontal. 
plt.xlabel("Temperatura (Â°C)") 

# Etiquetar el eje vertical. 
plt.ylabel("Respuesta del sensor (unidades arbitrarias)") 

# Mostrar la leyenda de la grÃ¡fica. 
plt.legend() 

# Ajustar el diseÃ±o para evitar superposiciones. 
plt.tight_layout() 

# Mostrar la grÃ¡fica. 
plt.show() 


#--------Paso5--------

# Seleccionar la temperatura como variable de entrada. 
X = datos[["temperatura"]] 

# Seleccionar la respuesta del sensor como variable objetivo. 
y = datos["respuesta_sensor"] 

# Reservar el 20 % de los registros para la evaluaciÃ³n. 
# El 80 % restante se utilizarÃ¡ para entrenar los modelos. 
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

   # Generar las caracterÃ­sticas de grado 2: x y xÂ². 
   # include_bias=False evita agregar una columna de unos, 
   # porque LinearRegression ya estima el intercepto. 
   ("polinomio", PolynomialFeatures( 
      degree=2, 
      include_bias=False 
    )), 

    # Ajustar una regresiÃ³n lineal sobre las caracterÃ­sticas generadas. 
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

# Calcular la raÃ­z del error cuadrÃ¡tico medio. 
rmse_grado_2 = np.sqrt(mean_squared_error(y_test, predicciones_grado_2)) 

# Calcular el coeficiente de determinaciÃ³n. 
r2_grado_2 = r2_score(y_test, predicciones_grado_2) 

# Imprimir las mÃ©tricas con cuatro decimales. 
print(f"MAE: {mae_grado_2:.4f}") 
print(f"RMSE: {rmse_grado_2:.4f}") 
print(f"RÂ²: {r2_grado_2:.4f}")


#--------Paso8--------

# Ordenar las temperaturas de prueba para dibujar una curva continua.
orden = np.argsort(X_test["temperatura"].to_numpy()) 

# Obtener las temperaturas de prueba en orden ascendente. 
temperaturas_ordenadas = X_test["temperatura"].to_numpy()[orden] 

# Reordenar las predicciones de acuerdo con la temperatura. 
predicciones_ordenadas = predicciones_grado_2[orden] 

# Crear la figura de comparaciÃ³n. 
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

# Agregar tÃ­tulo y etiquetas. 
plt.title("Ajuste de regresiÃ³n polinomial de grado 2") 
plt.xlabel("Temperatura (Â°C)") 
plt.ylabel("Respuesta del sensor") 

# Mostrar la leyenda. 
plt.legend() 

# Ajustar el diseÃ±o. 
plt.tight_layout() 

# Mostrar la grÃ¡fica.
plt.show()


#--------Paso9--------

# Definir los grados adicionales que se evaluaran.
grados_reto = [1, 2, 3, 4, 5, 6, 8, 10]

# Crear una lista para almacenar los resultados.
resultados_reto = []

# Crear un diccionario para almacenar los modelos entrenados.
modelos_reto = {}

# Recorrer los grados que se desean probar.
for grado in grados_reto:

    # Crear el pipeline para el grado actual.
    modelo = Pipeline([
        ("escalado", StandardScaler()),
        ("polinomio", PolynomialFeatures(
            degree=grado,
            include_bias=False
        )),
        ("regresion", LinearRegression())
    ])

    # Entrenar el modelo con los datos de entrenamiento.
    modelo.fit(X_train, y_train)

    # Obtener predicciones de entrenamiento.
    pred_train = modelo.predict(X_train)

    # Obtener predicciones de prueba.
    pred_test = modelo.predict(X_test)

    # Calcular el RMSE de entrenamiento.
    error_train = np.sqrt(
        mean_squared_error(y_train, pred_train)
    )

    # Calcular el RMSE de prueba.
    error_test = np.sqrt(
        mean_squared_error(y_test, pred_test)
    )

    # Calcular el R2 sobre el conjunto de prueba.
    r2_test = r2_score(y_test, pred_test)

    # Registrar las metricas del grado actual.
    resultados_reto.append({
        "Grado": grado,
        "RMSE entrenamiento": error_train,
        "RMSE prueba": error_test,
        "R2 prueba": r2_test
    })

    # Guardar el modelo entrenado para utilizarlo despues si se requiere.
    modelos_reto[grado] = modelo

# Convertir la lista en una tabla.
tabla_reto = pd.DataFrame(resultados_reto)

# Mostrar los resultados ordenados por grado.
print("\nResultados del reto ordenados por grado:")
print(tabla_reto.sort_values("Grado").round(4))

# Identificar el modelo con menor RMSE de prueba.
# Este criterio se usa aqui para explorar los resultados,
# pero no garantiza por si solo la mejor generalizacion futura.
indice_mejor = tabla_reto["RMSE prueba"].idxmin()

# Recuperar la fila correspondiente al menor RMSE de prueba.
mejor_resultado = tabla_reto.loc[indice_mejor]

# Mostrar el grado que obtuvo el menor RMSE de prueba.
print("\nGrado con menor RMSE de prueba:")
print(int(mejor_resultado["Grado"]))

# Mostrar las metricas del modelo seleccionado.
print("\nMetricas del modelo seleccionado:")
print(mejor_resultado.to_frame().T.round(4))

# Recuperar las metricas del modelo de grado 2 para compararlo.
resultado_grado_2 = tabla_reto.loc[
    tabla_reto["Grado"] == 2
].iloc[0]

# Comparar el mejor modelo contra el modelo de grado 2.
comparacion_reto = pd.DataFrame(
    [mejor_resultado, resultado_grado_2],
    index=["Mejor modelo", "Modelo grado 2"]
)

print("\nComparacion entre el mejor modelo y el modelo de grado 2:")
print(comparacion_reto.round(4))


#--------Paso10--------

# Crear una figura para comparar los errores.
plt.figure(figsize=(8, 5))

# Graficar el RMSE de entrenamiento en funcion del grado.
plt.plot(
    tabla_reto["Grado"],
    tabla_reto["RMSE entrenamiento"],
    marker="o",
    label="RMSE entrenamiento"
)

# Graficar el RMSE de prueba en funcion del grado.
plt.plot(
    tabla_reto["Grado"],
    tabla_reto["RMSE prueba"],
    marker="s",
    label="RMSE prueba"
)

# Etiquetar los ejes y agregar un titulo.
plt.title("Efecto del grado polinomial en el error")
plt.xlabel("Grado del polinomio")
plt.ylabel("RMSE")
plt.xticks(grados_reto)

# Mostrar la leyenda.
plt.legend()

# Ajustar el diseno.
plt.tight_layout()

# Mostrar la grafica.
plt.show()


#--------Paso11--------

# Identificar si el error de entrenamiento continua disminuyendo.
print("\nAnalisis del RMSE de entrenamiento:")
if tabla_reto["RMSE entrenamiento"].is_monotonic_decreasing:
    print("El RMSE de entrenamiento continua disminuyendo cuando aumenta el grado.")
else:
    print("El RMSE de entrenamiento no disminuye de forma continua en todos los grados.")

# Determinar la diferencia entre el mejor modelo y el modelo de grado 2.
grados_mejor = int(mejor_resultado["Grado"])
rmse_mejor = mejor_resultado["RMSE prueba"]
rmse_grado_2_reto = resultado_grado_2["RMSE prueba"]
mejora_relativa = (rmse_grado_2_reto - rmse_mejor) / rmse_grado_2_reto

print("\nJustificacion para el prototipo:")

if grados_mejor == 2:
    print(
        "Elegiria el modelo de grado 2 porque obtuvo el menor RMSE de prueba "
        "entre los modelos evaluados. Tambien es un modelo sencillo, por lo "
        "que reduce el riesgo de sobreajuste y facilita su implementacion."
    )
elif mejora_relativa < 0.05:
    print(
        f"El modelo con menor RMSE de prueba fue el de grado {grados_mejor}, "
        "pero su mejora frente al grado 2 fue menor al 5%. Por equilibrio "
        "entre precision, complejidad y menor riesgo de sobreajuste, elegiria "
        "el modelo de grado 2 para el prototipo."
    )
else:
    print(
        f"El modelo de grado {grados_mejor} obtuvo el menor RMSE de prueba y "
        "mejora de forma apreciable al modelo de grado 2. Lo elegiria para el "
        "prototipo, siempre revisando la grafica para confirmar que no haya "
        "senales claras de sobreajuste en los grados mas altos."
    )
