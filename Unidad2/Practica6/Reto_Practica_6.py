# 1. IMPORTACIÓN DE LIBRERÍAS

# Cargar el dataset
from sklearn.datasets import load_breast_cancer

# Dividir los datos
from sklearn.model_selection import train_test_split

# Árbol de decisión
from sklearn.tree import DecisionTreeClassifier

# Validación cruzada y búsqueda de parámetros
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Métricas
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# Crear tablas de resultados
import pandas as pd


# 2. CARGAR EL DATASET

# Cargar el conjunto de datos de cáncer de mama
datos = load_breast_cancer()

# Características de entrada
X = datos.data

# Variable objetivo
y = datos.target


# Mostrar información básica
print("Número de muestras:", X.shape[0])
print("Número de características:", X.shape[1])
print("Clases:", datos.target_names)


# 3. DIVIDIR LOS DATOS EN ENTRENAMIENTO Y PRUEBA

# 80% para entrenamiento y 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Mostrar tamaños
print("\nDatos de entrenamiento:", X_train.shape)
print("Datos de prueba:", X_test.shape)


# 4. EVALUAR DIFERENTES PROFUNDIDADES Y CRITERIOS

# Lista donde se guardarán los resultados
resultados = []

# Criterios que se compararán
criterios = ["gini", "entropy"]

# Profundidades solicitadas
profundidades = [2, 3, 4, 5, 6, None]


# Recorrer los dos criterios
for criterio in criterios:

    # Recorrer las diferentes profundidades
    for profundidad in profundidades:

        # Crear el árbol
        arbol = DecisionTreeClassifier(
            criterion=criterio,
            max_depth=profundidad,
            random_state=42
        )

        # Entrenar el árbol
        arbol.fit(X_train, y_train)

        # Predicciones sobre entrenamiento
        pred_train = arbol.predict(X_train)

        # Predicciones sobre prueba
        pred_test = arbol.predict(X_test)

        # Accuracy de entrenamiento
        accuracy_train = accuracy_score(
            y_train,
            pred_train
        )

        # F1 macro de entrenamiento
        f1_train = f1_score(
            y_train,
            pred_train,
            average="macro"
        )

        # Accuracy de prueba
        accuracy_test = accuracy_score(
            y_test,
            pred_test
        )

        # F1 macro de prueba
        f1_test = f1_score(
            y_test,
            pred_test,
            average="macro"
        )

        # Guardar todos los resultados
        resultados.append({
            "Criterio": criterio,
            "Profundidad máxima": profundidad,
            "Accuracy entrenamiento": accuracy_train,
            "F1 macro entrenamiento": f1_train,
            "Accuracy prueba": accuracy_test,
            "F1 macro prueba": f1_test,
            "Profundidad efectiva": arbol.get_depth(),
            "Número de hojas": arbol.get_n_leaves()
        })


# 5. MOSTRAR RESULTADOS DE LOS 12 MODELOS

# Convertir resultados a una tabla
resultados_df = pd.DataFrame(resultados)

print("\n")
print("=" * 100)
print("RESULTADOS DE LOS ÁRBOLES")
print("=" * 100)

print(
    resultados_df.round(4).to_string(index=False)
)


# 6. VALIDACIÓN CRUZADA PARA SELECCIONAR EL MEJOR MODELO

# Crear árbol base
arbol_base = DecisionTreeClassifier(
    random_state=42
)

# Parámetros que serán evaluados
parametros = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2, 3, 4, 5, 6, None],
    "min_samples_leaf": [1, 2, 4]
}


# Crear validación cruzada estratificada
validacion = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Configurar GridSearchCV
busqueda = GridSearchCV(
    estimator=arbol_base,
    param_grid=parametros,
    scoring="f1_macro",
    cv=validacion,
    n_jobs=-1,
    refit=True
)


# Realizar la búsqueda utilizando SOLO entrenamiento
busqueda.fit(X_train, y_train)


# 7. MOSTRAR LA MEJOR CONFIGURACIÓN

print("\n")
print("=" * 100)
print("MEJOR CONFIGURACIÓN ENCONTRADA")
print("=" * 100)

print("Mejores parámetros:")
print(busqueda.best_params_)

print("\nMejor F1-score macro de validación:")
print(round(busqueda.best_score_, 4))


# Obtener el mejor árbol
mejor_arbol = busqueda.best_estimator_


# 8. EVALUAR EL MEJOR MODELO EN ENTRENAMIENTO

# Predicciones de entrenamiento
pred_train_mejor = mejor_arbol.predict(X_train)

# Accuracy de entrenamiento
accuracy_train_mejor = accuracy_score(
    y_train,
    pred_train_mejor
)

# F1 macro de entrenamiento
f1_train_mejor = f1_score(
    y_train,
    pred_train_mejor,
    average="macro"
)



# 9. EVALUAR EL MEJOR MODELO EN PRUEBA

# Predicciones del conjunto de prueba
pred_mejor = mejor_arbol.predict(X_test)

# Accuracy de prueba
accuracy_mejor = accuracy_score(
    y_test,
    pred_mejor
)

# F1 macro de prueba
f1_mejor = f1_score(
    y_test,
    pred_mejor,
    average="macro"
)


# 10. MOSTRAR RESULTADOS DEL MEJOR MODELO

print("\n")
print("=" * 100)
print("RESULTADOS DEL MEJOR MODELO")
print("=" * 100)

print(
    f"Accuracy entrenamiento: "
    f"{accuracy_train_mejor:.4f}"
)

print(
    f"F1-score macro entrenamiento: "
    f"{f1_train_mejor:.4f}"
)

print(
    f"Accuracy prueba: "
    f"{accuracy_mejor:.4f}"
)

print(
    f"F1-score macro prueba: "
    f"{f1_mejor:.4f}"
)

print(
    "Profundidad efectiva:",
    mejor_arbol.get_depth()
)

print(
    "Número de hojas:",
    mejor_arbol.get_n_leaves()
)


# 11. REPORTE DE CLASIFICACIÓN

print("\n")
print("=" * 100)
print("REPORTE DE CLASIFICACIÓN")
print("=" * 100)

print(
    classification_report(
        y_test,
        pred_mejor,
        target_names=datos.target_names,
        zero_division=0
    )
)


# 12. COMPARACIÓN ENTRE ENTRENAMIENTO Y PRUEBA

# Calcular diferencia entre entrenamiento y prueba
diferencia_accuracy = (
    accuracy_train_mejor -
    accuracy_mejor
)

diferencia_f1 = (
    f1_train_mejor -
    f1_mejor
)


print("\n")
print("=" * 100)
print("COMPARACIÓN ENTRE ENTRENAMIENTO Y PRUEBA")
print("=" * 100)

print(
    f"Diferencia de Accuracy: "
    f"{diferencia_accuracy:.4f}"
)

print(
    f"Diferencia de F1-score macro: "
    f"{diferencia_f1:.4f}"
)


# 13. INTERPRETACIÓN DEL SOBREAJUSTE

if diferencia_f1 > 0.10:

    print(
        "\nSe observa una diferencia considerable entre "
        "el entrenamiento y la prueba, lo que puede indicar "
        "sobreajuste."
    )

else:

    print(
        "\nLa diferencia entre entrenamiento y prueba "
        "es relativamente pequeña, por lo que el modelo "
        "presenta un comportamiento estable."
    )


# 14. CONCLUSIÓN SOBRE LA COMPLEJIDAD

print("\n")
print("=" * 100)
print("CONCLUSIÓN")
print("=" * 100)

print(
    "La comparación de los modelos permite analizar cómo "
    "la profundidad del árbol afecta su desempeño. Los "
    "árboles con mayor profundidad pueden aprender patrones "
    "más complejos, aunque una complejidad excesiva puede "
    "provocar sobreajuste. Por ello, la configuración final "
    "se seleccionó mediante validación cruzada utilizando "
    "F1-score macro y posteriormente se evaluó sobre el "
    "conjunto de prueba reservado."
)