#-------- Paso 1 -------- 
# Importar las bibliotecas. 
# Importa NumPy para operaciones numéricas. 
import numpy as np 

# Importa Pandas para manipular datos tabulares. 
import pandas as pd 

# Importa Matplotlib para generar gráficas. 
import matplotlib.pyplot as plt 

# Importa Seaborn para visualizar matrices de confusión. 
import seaborn as sns 

# Importa la función para dividir datos de entrenamiento y prueba. 
from sklearn.model_selection import train_test_split, GridSearchCV 

# Importa un árbol de decisión para usarlo como referencia. 
from sklearn.tree import DecisionTreeClassifier 

# Importa el clasificador Bagging. 
from sklearn.ensemble import BaggingClassifier 

# Importa Random Forest. 
from sklearn.ensemble import RandomForestClassifier 

# Importa AdaBoost. 
from sklearn.ensemble import AdaBoostClassifier 

# Importa Gradient Boosting. 
from sklearn.ensemble import GradientBoostingClassifier 

# Importa las métricas de clasificación. 
from sklearn.metrics import ( 
   accuracy_score, 
   f1_score, 
   classification_report, 
   confusion_matrix 
)

#-------- Paso 2 --------
# Define la dirección del archivo CSV de vino tinto. 
url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv") 

# Lee el archivo CSV. 
# El separador utilizado en este conjunto de datos es ';'. 
df = pd.read_csv(url, sep=";") 

# Muestra las primeras cinco filas. 
print(df.head()) 

# Muestra el número de filas y columnas. 
print("Dimensiones:", df.shape) 

# Muestra los nombres de las columnas. 
print("Variables:", df.columns.tolist()) 

# Muestra información sobre tipos de datos y valores presentes. 
df.info() 

# Calcula estadísticas descriptivas. 
print(df.describe()) 

# Cuenta los valores faltantes en cada columna. 
print("Valores faltantes:") 
print(df.isnull().sum())

#-------- Paso 3 --------
# Crea una función para asignar una categoría a cada vino. 
def clasificar_calidad(valor): 
   # Asigna la categoría baja a calificaciones de 4 o menos. 
   if valor <= 4: 
      return 0 
   # Asigna la categoría media a calificaciones 5 y 6. 
   elif valor <= 6: 
      return 1 
   # Asigna la categoría alta a calificaciones de 7 o más. 
   else: 
      return 2 

# Aplica la función a la columna original de calidad. 
df["categoria"] = df["quality"].apply(clasificar_calidad) 

# Muestra la cantidad de vinos por categoría. 
print(df["categoria"].value_counts().sort_index()) 

# Crea una gráfica para observar la distribución de categorías. 
conteos = df["categoria"].value_counts().sort_index() 

# Establece el tamaño de la figura. 
plt.figure(figsize=(7, 4)) 

# Genera una gráfica de barras. 
plt.bar(["Baja", "Media", "Alta"], conteos.values) 

# Agrega el título. 
plt.title("Distribución de categorías de calidad") 

# Agrega la etiqueta del eje horizontal. 
plt.xlabel("Categoría") 

# Agrega la etiqueta del eje vertical. 
plt.ylabel("Número de muestras") 

# Ajusta la presentación. 
plt.tight_layout() 

# Muestra la gráfica. 
plt.show() 

#-------- Paso 4 --------
# Define las variables predictoras. 
# Se eliminan tanto la calidad original como la categoría objetivo. 
X = df.drop(columns=["quality", "categoria"]) 

# Define las etiquetas que se desean predecir. 
y = df["categoria"] 

# Divide los datos en 80% entrenamiento y 20% prueba. 
X_train, X_test, y_train, y_test = train_test_split( 
   X, 
   y, 
   test_size=0.20, 
   random_state=42, 
   stratify=y 
) 

# Muestra la cantidad de registros de entrenamiento. 
print("Entrenamiento:", X_train.shape) 

# Muestra la cantidad de registros de prueba. 
print("Prueba:", X_test.shape)

#-------- Paso 5 --------
# Define un diccionario que almacenará los modelos base. 
modelos = { 
   "Árbol de decisión": DecisionTreeClassifier( 
      max_depth=5, 
      random_state=42 
    ), 
    "Bagging": BaggingClassifier( 
       estimator=DecisionTreeClassifier( 
          max_depth=5, 
          random_state=42 
        ), 
       n_estimators=100, 
       random_state=42, 
       n_jobs=-1 
    ), 
    "Random Forest": RandomForestClassifier( 
       n_estimators=100, 
       max_depth=8, 
       random_state=42, 
       n_jobs=-1 
    ), 
    "AdaBoost": AdaBoostClassifier( 
       n_estimators=100, 
       learning_rate=0.5, 
       random_state=42 
    ), 
    "Gradient Boosting": GradientBoostingClassifier( 
       n_estimators=100, 
       learning_rate=0.1, 
       max_depth=3, 
       random_state=42 
    ) 
} 

# Recorre todos los modelos definidos. 
for nombre, modelo in modelos.items(): 
   modelo.fit(X_train, y_train) 
   print(f"Modelo entrenado: {nombre}") 

#-------- Paso 6 --------
# Crea una lista vacía para guardar los resultados. 
resultados = [] 

# Recorre cada modelo entrenado. 
for nombre, modelo in modelos.items(): 
   y_pred = modelo.predict(X_test) 
   accuracy = accuracy_score(y_test, y_pred) 
   f1_macro = f1_score( 
      y_test, 
      y_pred, 
      average="macro", 
      zero_division=0 
   ) 

   resultados.append({ 
      "Modelo": nombre, 
      "Accuracy": accuracy, 
      "F1_macro": f1_macro 
  }) 
   print(f"\nModelo: {nombre}") 
   print(classification_report( 
      y_test, 
      y_pred, 
      labels=[0, 1, 2], 
      target_names=["Baja", "Media", "Alta"], 
      zero_division=0 
)) 

#-------- RETO: Optimización de Random Forest con GridSearchCV --------
print("\n--- INICIO DE OPTIMIZACIÓN DE RANDOM FOREST ---")

# Define los hiperparámetros a evaluar mediante validación cruzada.
parametros = { 
    "n_estimators": [50, 100, 200], 
    "max_depth": [5, 10, None], 
    "min_samples_leaf": [1, 2, 4], 
    "class_weight": [None, "balanced"] 
} 

# Crea el modelo base de Random Forest para el grid search.
rf_base = RandomForestClassifier( 
    random_state=42, 
    n_jobs=-1 
) 

# Configura la búsqueda mediante validación cruzada enfocada en optimizar 'f1_macro'.
busqueda = GridSearchCV( 
    estimator=rf_base, 
    param_grid=parametros, 
    scoring="f1_macro", 
    cv=5, 
    n_jobs=-1 
) 

# Entrena y evalúa las combinaciones utilizando EXCLUSIVAMENTE los datos de entrenamiento.
busqueda.fit(X_train, y_train) 

# Muestra los mejores hiperparámetros encontrados.
print("Mejores parámetros encontrados:") 
print(busqueda.best_params_) 

# Muestra el mejor F1 macro promedio obtenido en validación cruzada.
print("Mejor F1 macro en validación cruzada:") 
print(busqueda.best_score_) 

# Obtiene el mejor estimador optimizado resultante de la búsqueda.
rf_optimizado = busqueda.best_estimator_

# Evalúa el desempeño final del modelo optimizado en el conjunto de prueba.
y_pred_opt = rf_optimizado.predict(X_test)
acc_opt = accuracy_score(y_test, y_pred_opt)
f1_opt = signal_f1 = f1_score(y_test, y_pred_opt, average="macro", zero_division=0)

# Agrega el Random Forest Optimizado a la lista de resultados para la comparación.
resultados.append({
    "Modelo": "Random Forest Optimizado",
    "Accuracy": acc_opt,
    "F1_macro": f1_opt
})

print("\nReporte de Clasificación: Random Forest Optimizado")
print(classification_report(
    y_test, 
    y_pred_opt, 
    labels=[0, 1, 2], 
    target_names=["Baja", "Media", "Alta"], 
    zero_division=0
))

#-------- Paso 7 (Actualizado): Comparación de Modelos --------
# Convierte los resultados a un DataFrame incluyendo el optimizado.
tabla_resultados = pd.DataFrame(resultados) 

# Ordena los modelos según su F1 macro. 
tabla_resultados = tabla_resultados.sort_values( 
   by="F1_macro", 
   ascending=False 
) 

print("\nTabla Comparativa Final:")
print(tabla_resultados)

# Crea una figura para comparar los modelos incluyendo el optimizado. 
plt.figure(figsize=(10, 5)) 

# Genera barras horizontales con los valores de F1 macro. 
plt.barh( 
   tabla_resultados["Modelo"], 
   tabla_resultados["F1_macro"],
   color=['skyblue' if m != 'Random Forest Optimizado' else 'orange' for m in tabla_resultados["Modelo"]]
)

plt.xlabel("F1-score macro") 
plt.title("Comparación de modelos de ensamble (Incluyendo Random Forest Optimizado)") 
plt.xlim(0, 1) 
plt.gca().invert_yaxis() 
plt.tight_layout() 
plt.show() 

#-------- Paso 8: Matriz de Confusión del Mejor Modelo Global --------
mejor_nombre = tabla_resultados.iloc[0]["Modelo"] 
print(f"\nEl mejor modelo global es: {mejor_nombre}")

# Selecciona el modelo correspondiente (si es el optimizado o uno previo)
if mejor_nombre == "Random Forest Optimizado":
    mejor_modelo = rf_optimizado
    y_pred_mejor = y_pred_opt
else:
    mejor_modelo = modelos[mejor_nombre] 
    y_pred_mejor = mejor_modelo.predict(X_test) 

matriz = confusion_matrix( 
   y_test, 
   y_pred_mejor, 
   labels=[0, 1, 2] 
) 

plt.figure(figsize=(6, 5)) 
sns.heatmap( 
   matriz, 
   annot=True, 
   fmt="d", 
   cmap="Blues", 
   xticklabels=["Baja", "Media", "Alta"], 
   yticklabels=["Baja", "Media", "Alta"] 
) 
plt.title(f"Matriz de confusión: {mejor_nombre}") 
plt.xlabel("Categoría predicha") 
plt.ylabel("Categoría real") 
plt.tight_layout() 
plt.show()

#-------- Paso 9: Importancia de Características del Modelo Optimizado --------
importancias = rf_optimizado.feature_importances_ 

df_importancia = pd.DataFrame({ 
  "Variable": X.columns, 
  "Importancia": importancias 
})

df_importancia = df_importancia.sort_values( 
   by="Importancia", 
   ascending=False 
) 

print("\nImportancia de características (Random Forest Optimizado):")
print(df_importancia.head(10)) 

top10 = df_importancia.head(10) 
plt.figure(figsize=(9, 5)) 
plt.barh( 
   top10["Variable"], 
   top10["Importancia"] 
)
plt.gca().invert_yaxis() 
plt.title("Importancia de características: Random Forest Optimizado") 
plt.xlabel("Importancia relativa") 
plt.tight_layout() 
plt.show()