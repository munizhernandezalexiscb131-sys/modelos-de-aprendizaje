#-------- Paso 1 --------
# Instala las bibliotecas necesarias para la práctica. 
# En Google Colab normalmente ya están disponibles. %pip install numpy pandas matplotlib seaborn scikit-learn 
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
from sklearn.model_selection import train_test_split 

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
url = ( "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv") 

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
# Define un diccionario que almacenará los modelos. 
modelos = { 
   # Árbol de decisión individual como modelo de referencia. 
   "Árbol de decisión": DecisionTreeClassifier( 
      max_depth=5, 
      random_state=42 
    ), 

    # Bagging combina varios árboles entrenados 
    # sobre muestras Bootstrap. 
    "Bagging": BaggingClassifier( 
       estimator=DecisionTreeClassifier( 
          max_depth=5, 
          random_state=42 
        ), 
       n_estimators=100, 
       random_state=42, 
       n_jobs=-1 
    ), 
    # Random Forest combina árboles y selección 
    # aleatoria de características. 
    "Random Forest": RandomForestClassifier( 
       n_estimators=100, 
       max_depth=8, 
       random_state=42, 
       n_jobs=-1 
    ), 
    # AdaBoost construye modelos secuenciales. 
    "AdaBoost": AdaBoostClassifier( 
       n_estimators=100, 
       learning_rate=0.5, 
       random_state=42 
    ), 
    # Gradient Boosting mejora progresivamente 
    # las predicciones mediante optimización de la pérdida. 
    "Gradient Boosting": GradientBoostingClassifier( 
       n_estimators=100, 
       learning_rate=0.1, 
       max_depth=3, 
       random_state=42 
    ) 
} 

# Recorre todos los modelos definidos. 
for nombre, modelo in modelos.items(): 
   # Entrena el modelo con los datos de entrenamiento. 
   modelo.fit(X_train, y_train) 
   # Informa al usuario que terminó el entrenamiento. 
   print(f"Modelo entrenado: {nombre}") 

#-------- Paso 6 --------
# Crea una lista vacía para guardar los resultados. 
resultados = [] 

# Recorre cada modelo entrenado. 
for nombre, modelo in modelos.items(): 
   
   # Obtiene las predicciones sobre el conjunto de prueba. 
   y_pred = modelo.predict(X_test) 
   
   # Calcula la exactitud. 
   accuracy = accuracy_score(y_test, y_pred) 

   # Calcula F1 macro, asignando el mismo peso a cada clase. 
   f1_macro = f1_score( 
      y_test, 
      y_pred, 
      average="macro", 
      zero_division=0 
   ) 

   # Guarda las métricas y el nombre del algoritmo. 
   resultados.append({ 
      "Modelo": nombre, 
      "Accuracy": accuracy, 
      "F1_macro": f1_macro 
  }) 
   # Muestra el reporte detallado por categoría. 
   print("\nModelo:", nombre) 
   print(classification_report( 
      y_test, 
      y_pred, 
      labels=[0, 1, 2], 
      target_names=["Baja", "Media", "Alta"], 
      zero_division=0 
)) 

# Convierte los resultados a un DataFrame. 
tabla_resultados = pd.DataFrame(resultados) 

# Ordena los modelos según su F1 macro. 
tabla_resultados = tabla_resultados.sort_values( 
   by="F1_macro", 
   ascending=False 
) 

# Muestra la tabla comparativa. 
print(tabla_resultados)

#-------- Paso 7 --------
# Crea una figura para comparar los modelos. 
plt.figure(figsize=(10, 5)) 

# Genera barras horizontales con los valores de F1 macro. 
plt.barh( 
   tabla_resultados["Modelo"], 
   tabla_resultados["F1_macro"] 
)

# Agrega la etiqueta del eje horizontal. 
plt.xlabel("F1-score macro") 

# Agrega un título. 
plt.title("Comparación de modelos de ensamble") 

# Establece el intervalo del eje horizontal. 
plt.xlim(0, 1) 

# Invierte el orden para colocar arriba el mejor modelo. 
plt.gca().invert_yaxis() 

# Ajusta los elementos gráficos. 
plt.tight_layout() 

# Muestra la gráfica. 
plt.show() 

#-------- Paso 8 --------
# Obtiene el nombre del modelo con mayor F1 macro. 
mejor_nombre = tabla_resultados.iloc[0]["Modelo"] 

# Recupera el objeto del modelo correspondiente. 
mejor_modelo = modelos[mejor_nombre] 

# Genera predicciones sobre los datos de prueba. 
y_pred_mejor = mejor_modelo.predict(X_test) 

# Calcula la matriz de confusión con un orden fijo de clases. 
matriz = confusion_matrix( 
   y_test, 
   y_pred_mejor, 
   labels=[0, 1, 2] 
) 

# Define el tamaño de la figura. 
plt.figure(figsize=(6, 5)) 

# Representa la matriz de confusión mediante un mapa de calor. 
sns.heatmap( 
   matriz, 
   annot=True, 
   fmt="d", 
   cmap="Blues", 
   xticklabels=["Baja", "Media", "Alta"], 
   yticklabels=["Baja", "Media", "Alta"] 
) 

# Agrega el título con el nombre del modelo. 
plt.title(f"Matriz de confusión: {mejor_nombre}") 

# Identifica el eje horizontal. 
plt.xlabel("Categoría predicha") 

# Identifica el eje vertical. 
plt.ylabel("Categoría real") 

# Ajusta la presentación. 
plt.tight_layout() 

# Muestra la figura. 
plt.show()

#-------- Paso 9 --------
# Recupera el modelo Random Forest previamente entrenado. 
modelo_rf = modelos["Random Forest"]

# Obtiene la importancia calculada para cada característica. 
importancias = modelo_rf.feature_importances_ 

# Crea una tabla con nombres de variables e importancias. 
df_importancia = pd.DataFrame({ 
  "Variable": X.columns, 
  "Importancia": importancias 
})

# Ordena las variables desde la más importante. 
df_importancia = df_importancia.sort_values( 
   by="Importancia", 
   ascending=False 
) 

# Muestra las diez características más importantes. 
print(df_importancia.head(10)) 

# Selecciona las diez características más importantes. 
top10 = df_importancia.head(10) 

# Define el tamaño de la figura. 
plt.figure(figsize=(9, 5)) 

# Genera una gráfica de barras horizontales. 
plt.barh( 
   top10["Variable"], 
   top10["Importancia"] 
)

# Invierte el eje vertical para mostrar primero la más importante. 
plt.gca().invert_yaxis() 

# Agrega el título. 
plt.title("Importancia de características: Random Forest") 

# Agrega la etiqueta del eje horizontal. 
plt.xlabel("Importancia relativa") 

# Ajusta la figura. 
plt.tight_layout() 

# Muestra la gráfica. 
plt.show()


# PARTE 1: RESULTADOS DE LA PRÁCTICA

# 1. Comparar el desempeño del árbol individual con los métodos de ensamble.
#Los ensambles ganan porque combinan varios árboles, siendo más estables y acertados.

# 2. Determinar si Bagging mejora los resultados del árbol de decisión.
#Sí, Bagging supera al árbol único al promediar múltiples muestras y reducir fallos.

# 3. Identificar el algoritmo con mayor F1-score macro.
#Gradient Boosting (o RF Optimizado) ofrece el mejor balance para todas las clases.

# 4. Analizar las categorías con mayor número de errores.
#"Baja" y "Alta" fallan más por tener pocas muestras; "Media" acumula errores por volumen.

# 5. Explicar qué variables fisicoquímicas presentan mayor importancia en Random Forest.
#El nivel de alcohol, sulfatos y acidez son las pistas principales para predecir la calidad.



# PARTE 2: CONCEPTOS Y TEORÍA

# 1. ¿Cuál es la principal diferencia entre Bagging y Boosting respecto a la forma en que se entrenan los modelos individuales?
#Bagging entrena árboles en paralelo (independientes); Boosting en serie (corrigiendo al anterior).

# 2. ¿Por qué un método de ensamble puede obtener mejores resultados que un árbol de decisión individual?
#Porque varias opiniones juntas compensan los errores individuales y evitan fallos drásticos.

# 3. ¿Qué diferencias se observaron entre Random Forest, AdaBoost y Gradient Boosting en la clasificación de calidad del vino?
#Random Forest es estable; Gradient Boosting equilibra clases difíciles; AdaBoost sufre con clases raras.

# 4. ¿Por qué resulta importante utilizar F1-score macro cuando existen diferencias entre las cantidades de muestras de cada categoría?
#Evalúa todas las clases por igual, evitando que la clase mayoritaria engañe la métrica.

# 5. ¿Qué efecto puede tener modificar los hiperparámetros n_estimators, max_depth y learning_rate sobre el desempeño y el sobreajuste de los modelos?
#Valores muy altos o mal calibrados provocan sobreajuste (memorización excesiva de los datos).