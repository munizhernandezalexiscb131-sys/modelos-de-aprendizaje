#-------- Paso 1 --------
# Importa NumPy para operaciones numéricas. 
import numpy as np 

# Importa Pandas para manipular conjuntos de datos. 
import pandas as pd 

# Importa Matplotlib para crear gráficas. 
import matplotlib.pyplot as plt 

# Importa Seaborn para visualizar datos. 
import seaborn as sns 

# Importa StandardScaler para estandarizar las variables. 
from sklearn.preprocessing import StandardScaler 

# Importa el algoritmo K-Means. 
from sklearn.cluster import KMeans 

# Importa la métrica de silueta. 
from sklearn.metrics import silhouette_score 

#-------- Paso 2 --------
# Define el nombre del archivo descargado desde Kaggle. 
ruta = "C:\\Users\\Alexis\\Documents\\Aprendizaje automatico\\Unidad2\\Practica8\\Mall_Customers.csv" 

# Carga el archivo CSV en un DataFrame. 
df = pd.read_csv(ruta) 

# Muestra las primeras cinco observaciones. 
print(df.head()) 

# Muestra las dimensiones del conjunto de datos. 
print("Dimensiones:", df.shape) 

# Muestra los nombres de las columnas. 
print("Columnas:", df.columns.tolist()) 

# Muestra los tipos de datos. 
df.info() 

# Obtiene estadísticas descriptivas. 
print(df.describe()) 

# Identifica los valores faltantes. 
print("Valores faltantes:") 
print(df.isnull().sum()) 

#-------- Paso 3 --------
# Selecciona las dos características originales del estudio. 
columnas = ["Annual Income (k$)", "Spending Score (1-100)"] 
# Extrae las columnas seleccionadas. 
X = df[columnas].copy() 

# Define el tamaño de la gráfica. 
plt.figure(figsize=(8, 6)) 

# Representa cada cliente mediante un punto. 
plt.scatter( 
  X["Annual Income (k$)"], 
  X["Spending Score (1-100)"], 
  alpha=0.7 
) 

# Agrega el título. 
plt.title("Distribución de clientes") 

# Identifica el eje horizontal. 
plt.xlabel("Ingreso anual (miles de dólares)") 

# Identifica el eje vertical. 
plt.ylabel("Puntuación de gasto") 

# Agrega una cuadrícula. 
plt.grid(alpha=0.3) 

# Ajusta los elementos gráficos. 
plt.tight_layout() 

# Muestra la gráfica. 
plt.show() 

#-------- Paso 4 --------
# Crea el objeto encargado de estandarizar los datos. 
escalador = StandardScaler() 

# Calcula la media y desviación estándar de cada variable. 
# Después transforma las observaciones. 
X_escalado = escalador.fit_transform(X) 

# Convierte los datos estandarizados en un DataFrame. 
df_escalado = pd.DataFrame( 
  X_escalado, 
  columns=columnas 
) 

# Muestra las primeras cinco observaciones transformadas. 
print(df_escalado.head()) 

# Muestra las medias de las variables escaladas. 
print("Medias:") 
print(df_escalado.mean()) 

# Muestra las desviaciones estándar poblacionales. 
print("Desviaciones estándar:") 
print(df_escalado.std(ddof=0)) 

#-------- Paso 5 --------
# Define el número inicial de clústeres. 
k = 5 

# Crea el modelo K-Means. 
modelo = KMeans( 
  n_clusters=k, 
  init="k-means++", 
  n_init=10, 
  random_state=42 
) 

# Entrena el modelo y obtiene el clúster asignado a cada cliente. 
etiquetas = modelo.fit_predict(X_escalado) 

# Crea una copia de los datos originales seleccionados. 
df_clusters = X.copy() 

# Agrega las etiquetas de agrupamiento. 
df_clusters["Cluster"] = etiquetas 

# Muestra las primeras observaciones clasificadas por grupo. 
print(df_clusters.head()) 

# Muestra la cantidad de clientes por clúster. 
print(df_clusters["Cluster"].value_counts().sort_index()) 

# Muestra la inercia obtenida. 
print("Inercia:", modelo.inertia_)

#-------- Paso 6 --------
# Recupera los centroides calculados en el espacio estandarizado. 
centroides_escalados = modelo.cluster_centers_ 

# Transforma los centroides a las unidades originales. 
centroides_originales = escalador.inverse_transform( 
  centroides_escalados 
) 

# Define el tamaño de la figura. 
plt.figure(figsize=(9, 6)) 

# Dibuja los clientes, utilizando un color por clúster. 
sns.scatterplot( 
  data=df_clusters, 
  x="Annual Income (k$)", 
  y="Spending Score (1-100)", 
  hue="Cluster", 
  palette="tab10", 
  s=70 
) 

# Dibuja los centroides con símbolos de cruz. 
plt.scatter( 
  centroides_originales[:, 0], 
  centroides_originales[:, 1], 
  marker="X", 
  s=250, 
  c="black", 
  label="Centroides" 
) 

# Agrega el título. 
plt.title("Segmentación de clientes mediante K-Means (2 Variables)") 

# Identifica el eje horizontal. 
plt.xlabel("Ingreso anual (miles de dólares)") 

# Identifica el eje vertical. 
plt.ylabel("Puntuación de gasto") 

# Muestra la leyenda. 
plt.legend() 

# Ajusta la presentación. 
plt.tight_layout() 

# Muestra la gráfica. 
plt.show()

#-------- Paso 7 --------
# Define los valores de K que serán evaluados. 
valores_k = range(1, 11) 

# Crea una lista para almacenar la inercia de cada modelo. 
inercias = [] 

# Recorre los valores de K. 
for k_actual in valores_k: 
  # Crea un modelo K-Means con el número actual de clústeres. 
  modelo_k = KMeans( 
    n_clusters=k_actual, 
    n_init=10, 
    random_state=42 
  ) 
  
  # Entrena el modelo utilizando los datos estandarizados. 
  modelo_k.fit(X_escalado) 
  
  # Guarda la inercia obtenida. 
  inercias.append(modelo_k.inertia_) 

# Crea una figura. 
plt.figure(figsize=(8, 5)) 

# Grafica la inercia en función del número de clústeres. 
plt.plot( 
  list(valores_k), 
  inercias, 
  marker="o" 
) 

# Agrega el título. 
plt.title("Método del codo") 

# Identifica el eje horizontal. 
plt.xlabel("Número de clústeres (K)") 

# Identifica el eje vertical. 
plt.ylabel("Inercia") 

# Muestra una cuadrícula. 
plt.grid(alpha=0.3) 

# Ajusta la presentación. 
plt.tight_layout() 

# Muestra la gráfica. 
plt.show()

#-------- Paso 8 --------
# Define los valores de K válidos para calcular la silueta. 
valores_k_silueta = range(2, 11) 

# Crea una lista para almacenar los resultados. 
siluetas = [] 

# Evalúa cada número de clústeres. 
for k_actual in valores_k_silueta: 

  # Construye un modelo con el valor actual de K. 
  modelo_k = KMeans( 
    n_clusters=k_actual, 
    n_init=10, 
    random_state=42 
  )

  # Entrena el modelo y obtiene las etiquetas. 
  etiquetas_k = modelo_k.fit_predict(X_escalado) 

  # Calcula el coeficiente promedio de silueta. 
  puntuacion = silhouette_score( 
    X_escalado, 
    etiquetas_k 
  ) 

  # Almacena el resultado. 
  siluetas.append(puntuacion) 

# Identifica la posición del valor máximo. 
indice_mejor = np.argmax(siluetas) 

# Recupera el valor de K correspondiente. 
mejor_k = list(valores_k_silueta)[indice_mejor] 

# Muestra el número de clústeres seleccionado. 
print("Mejor K según silueta (2 variables):", mejor_k) 

# Muestra la puntuación de silueta correspondiente. 
print("Coeficiente de silueta (2 variables):", siluetas[indice_mejor]) 


#-------- RETO: Ampliación de K-Means con tres características (Edad, Ingreso, Gasto) --------
print("\n--- INICIO DEL RETO: K-MEANS CON TRES CARACTERÍSTICAS ---")

# Define las tres características que se utilizarán en el reto. 
variables_reto = [ 
    "Age", 
    "Annual Income (k$)", 
    "Spending Score (1-100)" 
] 

# Extrae las columnas seleccionadas. 
X_reto = df[variables_reto].copy() 

# Crea un nuevo escalador para estandarizar las tres variables. 
escalador_reto = StandardScaler() 
X_reto_escalado = escalador_reto.fit_transform(X_reto) 

# Define los valores de K que se evaluarán. 
valores_reto = range(2, 11) 
resultados_reto = [] 

# Recorre cada valor de K para calcular el coeficiente de silueta. 
for k_actual in valores_reto: 
    modelo_reto = KMeans( 
        n_clusters=k_actual, 
        n_init=10, 
        random_state=42 
    ) 
    etiquetas_reto = modelo_reto.fit_predict(X_reto_escalado) 
    puntuacion_reto = silhouette_score(X_reto_escalado, etiquetas_reto) 
    resultados_reto.append(puntuacion_reto) 

# Identifica el mejor valor de K según la silueta con tres variables. 
mejor_k_reto = list(valores_reto)[np.argmax(resultados_reto)] 
print("Mejor K con tres variables:", mejor_k_reto) 
print("Mejor coeficiente de silueta con tres variables:", max(resultados_reto))

# Entrena el modelo final con el mejor K del reto. 
modelo_final_reto = KMeans( 
    n_clusters=mejor_k_reto, 
    n_init=10, 
    random_state=42 
) 
etiquetas_finales_reto = modelo_final_reto.fit_predict(X_reto_escalado) 

# Crea una copia del conjunto de datos original y asigna los nuevos clústeres del reto. 
df_reto_final = df.copy() 
df_reto_final["Cluster"] = etiquetas_finales_reto 

# Calcula el perfil de los clústeres (edad, ingresos, gasto y número de clientes). 
perfil_reto = df_reto_final.groupby("Cluster")[variables_reto].mean()
perfil_reto["Número de clientes"] = df_reto_final.groupby("Cluster").size() 

print("\nPerfil de los clústeres con tres variables:")
print(perfil_reto.round(2)) 

# Guarda los resultados en un archivo CSV específico para el reto. 
df_reto_final.to_csv("clientes_segmentados_tres_variables.csv", index=False) 
print("Archivo generado: clientes_segmentados_tres_variables.csv")