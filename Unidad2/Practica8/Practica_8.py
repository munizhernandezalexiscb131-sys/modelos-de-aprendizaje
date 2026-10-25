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
# Selecciona las dos características del estudio. 
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
k = 11 

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
plt.title("Segmentación de clientes mediante K-Means") 

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
print("Mejor K según silueta:", mejor_k) 

# Muestra la puntuación de silueta correspondiente. 
print("Coeficiente de silueta:", siluetas[indice_mejor]) 

# Crea la figura. 
plt.figure(figsize=(8, 5)) 

# Grafica el coeficiente de silueta. 
plt.plot( 
  list(valores_k_silueta), 
  siluetas, 
  marker="o" 
) 

# Agrega el título. 
plt.title("Coeficiente de silueta según K") 

# Identifica el eje horizontal. 
plt.xlabel("Número de clústeres (K)") 

# Identifica el eje vertical. 
plt.ylabel("Coeficiente de silueta") 

# Muestra una cuadrícula. 
plt.grid(alpha=0.3) 

# Ajusta la presentación. 
plt.tight_layout() 

# Muestra la gráfica. 
plt.show()

#-------- Paso 9 --------
# Crea el modelo con el mejor K según el coeficiente de silueta. 
modelo_final = KMeans( 
  n_clusters=mejor_k, 
  n_init=10, 
  random_state=42 
)

# Entrena el modelo y obtiene las etiquetas definitivas. 
etiquetas_finales = modelo_final.fit_predict( 
  X_escalado 
) 

# Crea una copia del conjunto de datos original. 
df_final = df.copy() 

# Agrega la etiqueta asignada a cada cliente. 
df_final["Cluster"] = etiquetas_finales 

# Calcula la cantidad de clientes de cada grupo. 
conteos_finales = df_final.groupby("Cluster").size() 

# Calcula los promedios de las características por grupo. 
perfil_clusters = df_final.groupby("Cluster")[columnas].mean()

# Agrega el tamaño de cada grupo. 
perfil_clusters["Número de clientes"] = conteos_finales 

# Muestra el perfil de los clústeres. 
print(perfil_clusters.round(2)) 

# Guarda los resultados en un archivo CSV. 
df_final.to_csv( 
  "clientes_segmentados.csv", 
  index=False 
) 

# Confirma que se generó el archivo. 
print("Archivo generado: clientes_segmentados.csv")



# ANÁLISIS DE LA PRÁCTICA 8 Y EL RETO

# 1. Identificar el valor de K sugerido por el método del codo.
#Sugiere K=5, ya que es el punto de inflexión donde la curva empieza a aplanarse.

# 2. Determinar el valor de K con mayor coeficiente de silueta.
#El mejor valor es K=5, obteniendo la mayor puntuación de separación entre grupos.

# 3. Comparar ambos resultados.
#Ambos métodos coinciden en que K=5 es la cantidad ideal para segmentar los clientes.

# 4. Explicar si los grupos encontrados tienen una interpretación práctica.
#Sí, permiten identificar claramente perfiles como clientes con ingresos altos o bajos y sus hábitos de gasto.

# 5. Analizar la influencia del escalamiento en el agrupamiento.
#Es vital para que variables con números más grandes no dominen o distorsionen los cálculos de distancia.



# PREGUNTAS DEL RETO (TRES VARIABLES)

# 1. ¿El número de clústeres seleccionado cambió al incorporar la edad?
#Sí, al añadir la edad el mejor valor de K pasó de 5 a 6 clústeres.

# 2. ¿Qué diferencias se observaron entre los grupos del modelo original y los del modelo ampliado?
#El modelo ampliado dividió mejor a los clientes considerando también su etapa de vida (jueventud o madurez).

# 3. ¿La incorporación de la edad permitió obtener segmentos más interpretables?
#Sí, porque ayuda a entender mejor los hábitos de compra según la edad del cliente.

# 4. ¿El coeficiente de silueta aumentó o disminuyó?
#Disminuyó ligeramente, ya que agregar más variables complica un poco la separación perfecta.

# 5. ¿Qué otras variables podrían ser útiles para mejorar la segmentación?
#Datos como género, frecuencia de visita al centro comercial o categoría de productos comprados.



# PREGUNTAS DE COMPRENSIÓN

# 1. ¿Cuál es la diferencia entre aprendizaje supervisado y aprendizaje no supervisado, y por qué K-Means pertenece al segundo?
#El supervisado usa datos etiquetados (respuestas conocidas); el no supervisado agrupa datos libres. K-Means es no supervisado porque descubre los grupos por sí solo.

# 2. ¿Qué representa un centroide y cómo se actualiza durante el funcionamiento del algoritmo K-Means?
#Es el punto central de un clúster; se recalcula moviéndose al promedio de todos los puntos asignados a ese grupo.

# 3. ¿Por qué es importante estandarizar las variables antes de aplicar K-Means?
#Porque K-Means se basa en distancias; si una variable usa números más grandes, afectará injustamente el resultado.

# 4. ¿Cuál es la diferencia entre el método del codo y el coeficiente de silueta para seleccionar el número de clústeres?
#El codo mide la reducción del error visualmente, mientras que la silueta mide numéricamente qué tan bien separados y definidos están los grupos.

# 5. ¿Qué limitaciones presenta K-Means cuando los grupos tienen formas irregulares, tamaños muy diferentes o contienen valores atípicos?
#Asume grupos circulares y tamaños similares, por lo que falla con formas extrañas o se distorsiona con datos extremos (atípicos).