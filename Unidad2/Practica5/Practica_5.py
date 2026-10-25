#--------paso 1--------

# Importar NumPy para operaciones numéricas. 
import numpy as np 

# Importar Pandas para organizar los datos en tablas. 
import pandas as pd 

# Importar Matplotlib para construir gráficas. 
import matplotlib.pyplot as plt 

# Importar el conjunto de datos de tumores de mama. 
from sklearn.datasets import load_breast_cancer 

# Importar la función para dividir los datos. 
from sklearn.model_selection import train_test_split 

# Importar StandardScaler para estandarizar las características. 
from sklearn.preprocessing import StandardScaler 

# Importar Pipeline para encadenar el escalado y el clasificador. 
from sklearn.pipeline import Pipeline 

# Importar el clasificador de máquinas de vectores de soporte. 
from sklearn.svm import SVC 

# Importar métricas para evaluar la clasificación. 
from sklearn.metrics import ( 
   accuracy_score, 
   precision_score, 
   recall_score, 
   f1_score, 
   confusion_matrix, 
   classification_report, 
   roc_auc_score, 
   ConfusionMatrixDisplay, 
   RocCurveDisplay 
) 


#--------paso 2--------

# Cargar los datos desde el módulo de conjuntos de datos de Scikit-learn. 
datos = load_breast_cancer() 

# Convertir las características en una tabla de Pandas. 
X = pd.DataFrame( 
   datos.data, 
   columns=datos.feature_names 
) 

# Convertir las etiquetas en una serie de Pandas. 
y = pd.Series( 
   datos.target, 
   name="diagnostico" 
) 

# Mostrar las primeras cinco observaciones. 
print(X.head())

# Mostrar las primeras cinco etiquetas. 
print(y.head()) 


#--------paso 3--------

# Mostrar el número de observaciones y características. 
print("Dimensiones de X:", X.shape) 

# Mostrar la cantidad de etiquetas disponibles. 
print("Cantidad de etiquetas:", y.shape[0]) 

# Mostrar las clases originales y sus códigos. 
print("Clases:", datos.target_names) 

# Contar las observaciones de cada clase. 
print("Distribución de las clases:") 
print(y.value_counts().sort_index()) 

# Calcular el porcentaje de observaciones por clase. 
print("Porcentaje por clase:") 
print((y.value_counts(normalize=True).sort_index() * 100).round(2)) 

# Contar los valores faltantes por característica. 
print("Valores faltantes por característica:") 
print(X.isnull().sum().sum())


#--------paso 4--------

# Contar las observaciones de cada clase. 
frecuencias = y.value_counts().sort_index() 

# Crear las etiquetas legibles para la gráfica. 
nombres_clases = [ 
  datos.target_names[0], 
  datos.target_names[1] 
] 

# Crear una figura para la gráfica. 
plt.figure(figsize=(7, 5)) 

# Dibujar las barras con las frecuencias de cada clase. 
plt.bar(nombres_clases, frecuencias.values) 

# Agregar el título de la gráfica. 
plt.title("Distribución de las clases del conjunto de datos") 

# Etiquetar el eje horizontal. 
plt.xlabel("Clase") 

# Etiquetar el eje vertical. 
plt.ylabel("Número de observaciones") 

# Ajustar los elementos de la figura. 
plt.tight_layout() 

# Mostrar la gráfica. 
plt.show()


#--------paso 5--------

# Dividir las características y las etiquetas en dos conjuntos. 
# El 80 % se utilizará para entrenar y el 20 % para probar. 
X_train, X_test, y_train, y_test = train_test_split( 
   X, 
   y, 
   test_size=0.20, 
   random_state=42, 
   stratify=y 
) 

# Mostrar el tamaño del conjunto de entrenamiento. 
print("Observaciones de entrenamiento:", X_train.shape[0]) 

# Mostrar el tamaño del conjunto de prueba. 
print("Observaciones de prueba:", X_test.shape[0]) 

# Comprobar la distribución de clases en entrenamiento. 
print("Clases en entrenamiento:") 
print(y_train.value_counts().sort_index()) 

# Comprobar la distribución de clases en prueba. 
print("Clases en prueba:") 
print(y_test.value_counts().sort_index())


#--------paso 6--------

# Crear una secuencia de procesamiento para el clasificador. 
modelo_lineal = Pipeline([ 
   # Estandarizar las características usando solo X_train. 
   ("escalado", StandardScaler()), 
   # Crear un SVM con frontera de decisión lineal. 
   # C controla el compromiso entre margen y errores de entrenamiento. 
   # probability=True habilita estimaciones de probabilidad, 
   # aunque requiere un ajuste adicional durante el entrenamiento. 
   ("clasificador", SVC( 
      kernel="linear", 
      C=1.0, 
      probability=True, 
      random_state=42 
    )) 
]) 

# Ajustar el escalador y entrenar el clasificador. 
modelo_lineal.fit(X_train, y_train) 

# Mostrar un mensaje para confirmar que la ejecución terminó. 
print("Modelo SVM lineal entrenado.") 


#--------paso 7--------

# Predecir las clases del conjunto de prueba. 
pred_lineal = modelo_lineal.predict(X_test) 

# Mostrar las primeras diez etiquetas reales. 
print("Etiquetas reales:") 
print(y_test.iloc[:10].to_list()) 

# Mostrar las primeras diez etiquetas predichas. 
print("Etiquetas predichas:") 
print(pred_lineal[:10])


#--------paso 8--------

# Calcular la proporción total de predicciones correctas. 
accuracy_lineal = accuracy_score(y_test, pred_lineal) 

# Calcular la precisión para la clase 0, que representa maligno. 
precision_lineal = precision_score( 
   y_test, 
   pred_lineal, 
   pos_label=0, 
   zero_division=0 
) 

# Calcular la sensibilidad para la clase 0, que representa maligno. 
recall_lineal = recall_score( 
   y_test, 
   pred_lineal, 
   pos_label=0, 
   zero_division=0 
) 

# Calcular el F1-score para la clase 0, que representa maligno. 
f1_lineal = f1_score( 
   y_test, 
   pred_lineal, 
   pos_label=0, 
   zero_division=0 
) 

# Mostrar los resultados con cuatro decimales. 
print(f"Accuracy: {accuracy_lineal:.4f}") 
print(f"Precisión para maligno: {precision_lineal:.4f}") 
print(f"Sensibilidad para maligno: {recall_lineal:.4f}") 
print(f"F1-score para maligno: {f1_lineal:.4f}")


#--------paso 9--------

# Calcular la matriz de confusión. 
matriz = confusion_matrix( 
   y_test, 
   pred_lineal, 
   labels=[0, 1] 
) 

# Crear la visualización de la matriz de confusión. 
visualizador = ConfusionMatrixDisplay( 
   confusion_matrix=matriz, 
   display_labels=["Maligno", "Benigno"] 
) 

# Dibujar la matriz con sus etiquetas. 
visualizador.plot() 

# Agregar el título de la gráfica. 
plt.title("Matriz de confusión del SVM lineal") 

# Ajustar el diseño. 
plt.tight_layout() 

# Mostrar la gráfica. 
plt.show() 


#--------paso 10--------

# Generar un reporte con precisión, sensibilidad y F1-score por clase. 
reporte_lineal = classification_report( 
   y_test, 
   pred_lineal, 
   labels=[0, 1], 
   target_names=["Maligno", "Benigno"], 
   zero_division=0 
) 

# Mostrar el reporte completo. 
print(reporte_lineal)


#--------paso 11--------

# Crear un pipeline para el clasificador con kernel RBF. 
modelo_rbf = Pipeline([ 
   # Estandarizar las características de entrada. 
   ("escalado", StandardScaler()), 
   # Configurar el SVM con una frontera potencialmente no lineal. 
   # C controla la penalización de los errores. 
   # gamma controla el alcance de la influencia de las observaciones. 
   ("clasificador", SVC( 
       kernel="rbf", 
       C=1.0, 
       gamma="scale", 
       probability=True, 
       random_state=42 
    )) 
]) 

# Entrenar el modelo RBF con los datos de entrenamiento. 
modelo_rbf.fit(X_train, y_train) 

# Predecir las etiquetas del conjunto de prueba. 
pred_rbf = modelo_rbf.predict(X_test) 

# Mostrar un mensaje para identificar el modelo entrenado. 
print("Modelo SVM con kernel RBF entrenado.")


#--------paso 12--------

# Calcular las métricas del modelo lineal. 
metricas_lineal = { 
   "Modelo": "SVM lineal", 
   "Accuracy": accuracy_score(y_test, pred_lineal), 
   "Precisión maligno": precision_score(y_test, pred_lineal, pos_label=0, zero_division=0), 
   "Sensibilidad maligno": recall_score(y_test, pred_lineal, pos_label=0, zero_division=0), 
   "F1 maligno": f1_score(y_test, pred_lineal, pos_label=0, zero_division=0 ) 
} 

# Calcular las métricas del modelo RBF. 
metricas_rbf = { 
   "Modelo": "SVM RBF", 
   "Accuracy": accuracy_score(y_test, pred_rbf), 
   "Precisión maligno": precision_score(y_test, pred_rbf, pos_label=0, zero_division=0), 
   "Sensibilidad maligno": recall_score(y_test, pred_rbf, pos_label=0, zero_division=0), 
   "F1 maligno": f1_score(y_test, pred_rbf, pos_label=0, zero_division=0) 
} 

# Organizar las métricas en una tabla comparativa. 
tabla_comparacion = pd.DataFrame([ 
   metricas_lineal, 
   metricas_rbf 
]) 

# Mostrar la tabla redondeada a cuatro decimales. 
print(tabla_comparacion.round(4)) 


#--------paso 13--------

# Obtener las probabilidades estimadas para cada clase del modelo lineal. 
prob_lineal = modelo_lineal.predict_proba(X_test) 

# Obtener las probabilidades estimadas para cada clase del modelo RBF. 
prob_rbf = modelo_rbf.predict_proba(X_test) 

# Identificar la posición de la clase maligna en el orden de clases del modelo.
indice_maligno_lineal = list( 
   modelo_lineal.named_steps["clasificador"].classes_ 
).index(0) 

# Identificar la posición de la clase maligna en el modelo RBF. 
indice_maligno_rbf = list( 
   modelo_rbf.named_steps["clasificador"].classes_ 
).index(0) 

# Extraer las probabilidades de pertenecer a la clase maligna. 
prob_maligno_lineal = prob_lineal[:, indice_maligno_lineal] 
prob_maligno_rbf = prob_rbf[:, indice_maligno_rbf] 

# Construir una etiqueta binaria: 1 para maligno y 0 para benigno. 
y_test_maligno = (y_test == 0).astype(int) 

# Calcular ROC-AUC para las probabilidades de la clase maligna. 
auc_lineal = roc_auc_score( 
   y_test_maligno, 
   prob_maligno_lineal 
) 

# Calcular ROC-AUC del modelo RBF. 
auc_rbf = roc_auc_score( 
   y_test_maligno, 
   prob_maligno_rbf 
) 

# Mostrar ambos resultados. 
print(f"ROC-AUC SVM lineal: {auc_lineal:.4f}") 
print(f"ROC-AUC SVM RBF: {auc_rbf:.4f}") 


#--------paso 14--------

# Crear una figura para comparar las curvas ROC. 
fig, ax = plt.subplots(figsize=(7, 6)) 

# Dibujar la curva ROC del modelo lineal. 
RocCurveDisplay.from_predictions( 
   y_test_maligno, 
   prob_maligno_lineal, 
   name="SVM lineal", 
   ax=ax 
) 

# Dibujar la curva ROC del modelo RBF. 
RocCurveDisplay.from_predictions( 
   y_test_maligno, 
   prob_maligno_rbf, 
   name="SVM RBF", 
   ax=ax 
) 

# Agregar el título de la figura. 
ax.set_title("Comparación de curvas ROC") 

# Ajustar el diseño. 
plt.tight_layout() 

# Mostrar la figura. 
plt.show() 