#-------- Paso 1 --------
# Importar NumPy para realizar operaciones numéricas. 
import numpy as np 

# Importar Pandas para organizar los datos en tablas. 
import pandas as pd 

# Importar Matplotlib para crear visualizaciones. 
import matplotlib.pyplot as plt

# Importar el conjunto de datos Wine desde Scikit-learn. 
from sklearn.datasets import load_wine 

# Importar la función para dividir los datos. 
from sklearn.model_selection import train_test_split 

# Importar el algoritmo de árbol de decisión. 
from sklearn.tree import DecisionTreeClassifier 

# Importar la función para visualizar la estructura del árbol. 
from sklearn.tree import plot_tree 

# Importar métricas de evaluación para clasificación. 
from sklearn.metrics import ( 
   accuracy_score, 
   classification_report, 
   confusion_matrix, 
   ConfusionMatrixDisplay, 
   f1_score 
) 


#-------- Paso 2 --------
# Cargar el conjunto de datos Wine. 
datos = load_wine() 

# Convertir la matriz de características en un DataFrame. 
# Cada columna tendrá el nombre de una característica química. 
X = pd.DataFrame( 
   datos.data, 
   columns=datos.feature_names 
) 

# Convertir las etiquetas numéricas en una serie. 
y = pd.Series( 
   datos.target, 
   name="clase" 
) 

# Mostrar las primeras cinco observaciones. 
print(X.head()) 

# Mostrar las primeras cinco etiquetas. 
print(y.head()) 

# Mostrar los nombres de las tres clases. 
print("Clases:", datos.target_names) 


#-------- Paso 3 --------
# Mostrar el número de observaciones y características. 
print("Dimensiones de X:", X.shape) 

# Mostrar la cantidad de etiquetas. 
print("Cantidad de etiquetas:", len(y)) 

# Mostrar los nombres de las características. 
print("Características disponibles:") 
print(X.columns.tolist()) 

# Calcular estadísticas descriptivas de las variables. 
print(X.describe()) 

# Contar los valores faltantes de cada característica. 
print("Valores faltantes por característica:") 
print(X.isnull().sum()) 

# Contar cuántas observaciones existen por clase. 
print("Distribución de clases:") 
print(y.value_counts().sort_index()) 

#-------- Paso 4 --------
# Contar las observaciones de cada clase en orden numérico. 
frecuencias = y.value_counts().sort_index() 

# Obtener los nombres de las clases en el orden de sus etiquetas. 
nombres_clases = datos.target_names 

# Crear una figura para la gráfica. 
plt.figure(figsize=(7, 5)) 

# Dibujar una barra por cada clase. 
plt.bar(nombres_clases, frecuencias.values) 

# Agregar el título de la gráfica. 
plt.title("Distribución de las clases de vino") 

# Etiquetar el eje horizontal. 
plt.xlabel("Clase") 

# Etiquetar el eje vertical. 
plt.ylabel("Número de observaciones") 

# Ajustar el diseño para evitar superposiciones. 
plt.tight_layout() 

# Mostrar la gráfica. 
plt.show()

#-------- Paso 5 --------
# Separar las características y las etiquetas en entrenamiento y prueba. 
# El 80 % se utilizará para entrenar y el 20 % para evaluar. 
X_train, X_test, y_train, y_test = train_test_split( 
   X, 
   y, 
   test_size=0.20, 
   random_state=42, 
   stratify=y 
) 

# Mostrar la cantidad de muestras de entrenamiento. 
print("Muestras de entrenamiento:", X_train.shape[0]) 

# Mostrar la cantidad de muestras de prueba. 
print("Muestras de prueba:", X_test.shape[0]) 

# Mostrar la distribución de clases en entrenamiento. 
print("Clases en entrenamiento:") 
print(y_train.value_counts().sort_index()) 

# Mostrar la distribución de clases en prueba. 
print("Clases en prueba:") 
print(y_test.value_counts().sort_index()) 

#-------- Paso 6 --------
# Crear el árbol de decisión. 
# criterion="gini" utiliza la impureza de Gini para elegir las divisiones. 
# max_depth=3 limita la profundidad para facilitar la interpretación. 
# min_samples_leaf=2 exige al menos dos observaciones por hoja. 
# random_state=42 permite reproducir el entrenamiento. 
modelo = DecisionTreeClassifier( 
   criterion="gini", 
   max_depth=3, 
   min_samples_leaf=2, 
   random_state=42 
) 

# Entrenar el árbol con las características y etiquetas de entrenamiento. 
modelo.fit(X_train, y_train) 

# Mostrar un mensaje al finalizar el entrenamiento. 
print("Árbol de decisión entrenado correctamente.") 

#-------- Paso 7 --------
# Utilizar el árbol entrenado para predecir las clases de prueba. 
y_pred = modelo.predict(X_test) 

# Crear una tabla con las etiquetas reales y las predichas. 
resultados = pd.DataFrame({ 
   "Clase real": y_test, 
   "Clase predicha": y_pred 
}) 

# Mostrar las primeras diez comparaciones. 
print(resultados.head(10))

#-------- Paso 8 --------
# Calcular la proporción de predicciones correctas. 
accuracy = accuracy_score(y_test, y_pred) 

# Calcular el F1-score macro. 
# Este promedio da el mismo peso a cada clase. 
f1_macro = f1_score( 
   y_test, 
   y_pred, 
   average="macro" 
) 

# Mostrar la exactitud general. 
print(f"Accuracy: {accuracy:.4f}") 

# Mostrar el F1-score macro. 
print(f"F1-score macro: {f1_macro:.4f}") 

# Mostrar precisión, recall y F1-score de cada clase. 
print( 
   classification_report( 
      y_test, 
      y_pred, 
      target_names=datos.target_names, 
      zero_division=0 
    ) 
)

#-------- Paso 9 --------
# Calcular la matriz de confusión usando el orden de las clases. 
matriz = confusion_matrix( 
   y_test, 
   y_pred, 
   labels=[0, 1, 2] 
) 

# Crear el objeto que permite visualizar la matriz. 
visualizador = ConfusionMatrixDisplay( 
   confusion_matrix=matriz, 
   display_labels=datos.target_names 
) 

# Dibujar la matriz de confusión. 
visualizador.plot() 

# Agregar un título a la figura. 
plt.title("Matriz de confusión del árbol de decisión") 

# Ajustar el diseño. 
plt.tight_layout() 

# Mostrar la figura. 
plt.show() 

#-------- Paso 10 --------
# Crear una figura suficientemente grande para leer las reglas. 
plt.figure(figsize=(18, 10)) 

# Dibujar el árbol entrenado. 
plot_tree( 
   modelo, 
   feature_names=datos.feature_names, 
   class_names=datos.target_names, 
   filled=True, 
   rounded=True, 
   fontsize=9 
) 

# Agregar un título a la figura. 
plt.title("Estructura del árbol de decisión") 

# Ajustar el diseño. 
plt.tight_layout() 

# Mostrar el árbol. 
plt.show() 

#-------- Paso 11 --------
# Obtener la importancia asignada a cada característica por el árbol. 
importancias = modelo.feature_importances_ 

# Crear una tabla con los nombres y las importancias. 
tabla_importancias = pd.DataFrame({ 
   "Característica": X.columns, 
   "Importancia": importancias 
})

# Ordenar las características de mayor a menor importancia. 
tabla_importancias = tabla_importancias.sort_values( 
   by="Importancia", 
   ascending=False 
) 
# Mostrar las diez características con mayor importancia. 
print(tabla_importancias.head(10)) 

# Seleccionar las diez características principales para la gráfica. 
top_importancias = tabla_importancias.head(10).sort_values( 
   by="Importancia" 
) 

# Crear una figura horizontal. 
plt.figure(figsize=(9, 6)) 

# Dibujar las importancias de las diez características principales. 
plt.barh( 
   top_importancias["Característica"], 
   top_importancias["Importancia"] 
) 

# Agregar título y etiquetas. 
plt.title("Importancia de las características") 
plt.xlabel("Importancia asignada por el árbol") 
plt.ylabel("Característica") 

# Ajustar el diseño. 
plt.tight_layout() 

# Mostrar la gráfica. 
plt.show() 

# 1. ¿Qué característica aparece en la raíz del árbol y qué condición utiliza?
# Aparece la variable proline como división principal, evaluando si su valor es menor o igual a un umbral específico.

# 2. ¿Qué clases se confunden con mayor frecuencia en la matriz de confusión?
# La clase 1 y la clase 2, debido a que sus métricas químicas presentan una similitud considerable.

# 3. ¿Qué características tienen mayor importancia según el árbol entrenado?
# Las ubicadas en los niveles superiores del modelo, las cuales separan las observaciones con mayor eficacia.

# 4. ¿Cómo influye limitar la profundidad en la interpretabilidad del modelo?
# Simplifica la estructura del árbol, permitiendo leer e interpretar sus reglas de decisión sin complicaciones.

# 5. ¿Por qué las reglas aprendidas no deben interpretarse automáticamente como relaciones causales?
# Porque el algoritmo solo identifica asociaciones o tendencias numéricas entre datos, no causas reales.

# 6. ¿Cuáles son los componentes principales de un árbol de decisión y qué función cumple cada uno?
# Raíz (comienza el filtrado), nodos internos (aplican condiciones) y hojas (definen la categoría final).

# 7. ¿Qué diferencia existe entre los criterios de Gini y entropía?
# Ambos calculan la mezcla de datos, pero Gini mide probabilidades de error y Entropía cuantifica el desorden.

# 8. ¿Cómo puede afectar la profundidad del árbol al subajuste y al sobreajuste?
# Poca profundidad impide al modelo aprender; demasiada hace que memorice los datos y falle ante muestras nuevas.

# 9. ¿Qué información proporciona la importancia de las características y cuáles son sus limitaciones?
# Refleja qué variables fueron más útiles, aunque si dos son parecidas, asignará el crédito a una sola.

# 10. ¿Por qué es conveniente utilizar validación cruzada para seleccionar los parámetros de un árbol de decisión?
# Garantiza probar configuraciones sobre diferentes cortes de datos, evitando elegir parámetros con sesgo.