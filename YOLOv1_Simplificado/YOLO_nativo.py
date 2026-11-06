# ============================================
# YOLO NATIVO 6.1
# Detector integrado con fusion sin cadenas e IoU/NMS
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
import sys


# ============================================
# 1. CARGAR IMAGEN
# ============================================

def cargar_imagen(ruta):

    imagen = Image.open(ruta).convert("RGB")
    imagen.thumbnail((600, 600))

    return np.array(imagen)


# ============================================
# 2. CONVERTIR A ESCALA DE GRISES
# ============================================

def escala_grises(imagen):

    gris = (
        0.299 * imagen[:, :, 0] +
        0.587 * imagen[:, :, 1] +
        0.114 * imagen[:, :, 2]
    )

    return gris


# ============================================
# 3. DETECCION DE BORDES SOBEL
# ============================================

def detectar_bordes(gris):

    imagen = np.pad(gris, 1, mode="edge")

    gx = (
        -imagen[:-2, :-2] + imagen[:-2, 2:]
        - 2 * imagen[1:-1, :-2]
        + 2 * imagen[1:-1, 2:]
        - imagen[2:, :-2] + imagen[2:, 2:]
    )

    gy = (
        -imagen[:-2, :-2]
        - 2 * imagen[:-2, 1:-1]
        - imagen[:-2, 2:]
        + imagen[2:, :-2]
        + 2 * imagen[2:, 1:-1]
        + imagen[2:, 2:]
    )

    magnitud = np.sqrt(gx**2 + gy**2)

    umbral = np.percentile(magnitud, 85)

    bordes = magnitud > umbral

    return bordes


# ============================================
# 4. DILATACION DE BORDES
# ============================================

def dilatar(mascara, iteraciones=2):

    resultado = mascara.copy()

    for _ in range(iteraciones):

        anterior = np.pad(resultado, 1)

        nuevo = np.zeros_like(resultado)

        for y in range(3):
            for x in range(3):

                nuevo |= anterior[
                    y:y + resultado.shape[0],
                    x:x + resultado.shape[1]
                ]

        resultado = nuevo

    return resultado


# ============================================
# 5. BUSCAR COMPONENTES CONECTADOS
# ============================================

def componentes_conectados(mascara):

    alto, ancho = mascara.shape

    visitados = np.zeros_like(mascara, dtype=bool)

    componentes = []

    vecinos = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for y in range(alto):

        for x in range(ancho):

            if mascara[y, x] and not visitados[y, x]:

                cola = deque([(y, x)])

                visitados[y, x] = True

                puntos = []

                while cola:

                    cy, cx = cola.popleft()

                    puntos.append((cy, cx))

                    for dy, dx in vecinos:

                        ny = cy + dy
                        nx = cx + dx

                        if (
                            0 <= ny < alto and
                            0 <= nx < ancho and
                            mascara[ny, nx] and
                            not visitados[ny, nx]
                        ):

                            visitados[ny, nx] = True

                            cola.append((ny, nx))

                componentes.append(puntos)

    return componentes


# ============================================
# 6. GENERAR CAJAS DELIMITADORAS
# ============================================

def generar_cajas(componentes, alto, ancho):

    cajas = []

    for puntos in componentes:

        if len(puntos) < 40:
            continue

        ys = [p[0] for p in puntos]
        xs = [p[1] for p in puntos]

        x1 = min(xs)
        y1 = min(ys)

        x2 = max(xs)
        y2 = max(ys)

        area = (x2 - x1) * (y2 - y1)

        proporcion = area / (alto * ancho)

        if 0.005 < proporcion < 0.80:

            cajas.append([x1, y1, x2, y2])

    return cajas


# ============================================
# 7. CALCULAR DISTANCIA ENTRE CAJAS
# ============================================

def distancia_cajas(a, b):

    dx = max(a[0] - b[2], b[0] - a[2], 0)

    dy = max(a[1] - b[3], b[1] - a[3], 0)

    distancia = np.sqrt(dx**2 + dy**2)

    return distancia


# ============================================
# 8. AGRUPAR CAJAS CERCANAS
# ============================================

def agrupar_cajas(cajas, distancia_maxima):

    grupos = []

    # Cada caja debe estar cerca de todas las cajas del grupo.
    # Esto evita que una cadena de cajas pequeñas una regiones lejanas.
    for caja in cajas:

        grupo_cercano = None

        for grupo in grupos:

            distancias = [
                distancia_cajas(caja, caja_grupo)
                for caja_grupo in grupo
            ]

            if all(distancia <= distancia_maxima for distancia in distancias):
                grupo_cercano = grupo
                break

        if grupo_cercano is None:
            grupos.append([caja])
        else:
            grupo_cercano.append(caja)

    cajas_finales = []

    for grupo in grupos:

        x1 = min(c[0] for c in grupo)
        y1 = min(c[1] for c in grupo)

        x2 = max(c[2] for c in grupo)
        y2 = max(c[3] for c in grupo)

        cajas_finales.append([x1, y1, x2, y2])

    return cajas_finales


# ============================================
# 9. PROPUESTAS MULTIESCALA CON VENTANA DESLIZANTE
# ============================================

def generar_propuestas(bordes, maximo=80):
    """Busca ventanas horizontales con actividad de bordes y contorno."""

    alto, ancho = bordes.shape
    propuestas = []
    tamanos = [
        (0.42, 0.28), (0.52, 0.32), (0.62, 0.36),
        (0.72, 0.40), (0.62, 0.48)
    ]

    for proporcion_ancho, proporcion_alto in tamanos:
        ventana_ancho = int(ancho * proporcion_ancho)
        ventana_alto = int(alto * proporcion_alto)
        paso_x = max(1, ventana_ancho // 8)
        paso_y = max(1, ventana_alto // 8)

        for y1 in range(0, alto - ventana_alto + 1, paso_y):
            for x1 in range(0, ancho - ventana_ancho + 1, paso_x):
                x2 = x1 + ventana_ancho
                y2 = y1 + ventana_alto
                region = bordes[y1:y2, x1:x2]
                grosor = max(1, min(5, ventana_alto // 20))
                contorno = np.concatenate((
                    region[:grosor, :].ravel(),
                    region[-grosor:, :].ravel(),
                    region[:, :grosor].ravel(),
                    region[:, -grosor:].ravel()
                ))

                densidad = float(np.mean(region))
                densidad_contorno = float(np.mean(contorno))
                puntaje = 0.55 * min(densidad / 0.12, 1.0) + 0.45 * densidad_contorno

                propuestas.append({
                    "caja": [x1, y1, x2, y2],
                    "puntaje": puntaje
                })

    propuestas.sort(key=lambda propuesta: propuesta["puntaje"], reverse=True)
    return propuestas[:maximo]


# ============================================
# 9. EXTRAER CARACTERISTICAS
# ============================================

def extraer_caracteristicas(caja, bordes):

    x1, y1, x2, y2 = caja

    ancho = x2 - x1
    alto = y2 - y1

    if ancho == 0 or alto == 0:
        return None

    # Area relativa
    area = (ancho * alto) / bordes.size

    # Relacion entre ancho y alto
    aspecto = ancho / alto

    # Densidad de bordes
    region = bordes[y1:y2, x1:x2]

    densidad = np.mean(region)

    return {
        "area": area,
        "aspecto": aspecto,
        "densidad": densidad
    }


# ============================================
# 10. CLASIFICAR REGIONES
# ============================================

def clasificar_region(caracteristicas):

    if caracteristicas is None:

        return "Descartado", 0

    area = caracteristicas["area"]

    aspecto = caracteristicas["aspecto"]

    densidad = caracteristicas["densidad"]

    confianza = 0

    # Evaluar tamaño
    if 0.02 <= area <= 0.80:

        confianza += 0.30

    # Evaluar forma horizontal
    if 1.2 <= aspecto <= 4.5:

        confianza += 0.40

    # Evaluar cantidad de bordes
    if 0.03 <= densidad <= 0.70:

        confianza += 0.30

    if confianza >= 0.60:

        return "Objeto candidato", confianza

    return "Fondo o region descartada", confianza


# ============================================
# 11. ANALIZAR TODAS LAS CAJAS
# ============================================

def analizar_cajas(cajas, bordes):

    resultados = []

    for caja in cajas:

        caracteristicas = extraer_caracteristicas(
            caja,
            bordes
        )

        clase, confianza = clasificar_region(
            caracteristicas
        )

        resultados.append({

            "caja": caja,

            "clase": clase,

            "confianza": confianza,

            "caracteristicas": caracteristicas

        })

    return resultados


# ============================================
# 12. IoU Y SUPRESION DE CAJAS DUPLICADAS (NMS)
# ============================================

def calcular_iou(caja_a, caja_b):

    ax1, ay1, ax2, ay2 = caja_a
    bx1, by1, bx2, by2 = caja_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    ancho_interseccion = max(0, inter_x2 - inter_x1)
    alto_interseccion = max(0, inter_y2 - inter_y1)
    area_interseccion = ancho_interseccion * alto_interseccion

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - area_interseccion

    if union == 0:
        return 0.0

    return area_interseccion / union


def supresion_no_maxima(resultados, umbral_iou=0.20, maximo=3):
    """Conserva las mejores cajas y elimina propuestas redundantes."""

    candidatos = [
        resultado for resultado in resultados
        if resultado["clase"] == "Objeto candidato"
    ]

    candidatos.sort(
        key=lambda resultado: resultado.get("puntaje", resultado["confianza"]),
        reverse=True
    )
    conservados = []

    while candidatos:
        mejor = candidatos.pop(0)
        conservados.append(mejor)

        if len(conservados) >= maximo:
            break

        candidatos = [
            resultado for resultado in candidatos
            if calcular_iou(mejor["caja"], resultado["caja"]) <= umbral_iou
        ]

    return conservados


def fusionar_cajas_solapadas(resultados, bordes):
    """Fusiona cajas mutuamente solapadas sin unir cadenas de objetos."""

    grupos = []

    # Una caja se une a un grupo solo si se solapa con cada miembro.
    # Asi, una propuesta puente no fusiona objetos distintos en cadena.
    for indice, resultado in enumerate(resultados):
        grupo_destino = None

        for grupo in grupos:
            solapa_con_todos = all(
                calcular_iou(
                    resultado["caja"],
                    resultados[indice_grupo]["caja"]
                ) > 0
                for indice_grupo in grupo
            )

            if solapa_con_todos:
                grupo_destino = grupo
                break

        if grupo_destino is None:
            grupos.append([indice])
        else:
            grupo_destino.append(indice)

    fusionados = []

    for indices in grupos:
        detecciones = [resultados[i] for i in indices]
        caja = [
            min(d["caja"][0] for d in detecciones),
            min(d["caja"][1] for d in detecciones),
            max(d["caja"][2] for d in detecciones),
            max(d["caja"][3] for d in detecciones)
        ]
        caracteristicas = extraer_caracteristicas(caja, bordes)
        clase, confianza = clasificar_region(caracteristicas)

        fusionados.append({
            "caja": caja,
            "clase": clase,
            "confianza": confianza,
            "caracteristicas": caracteristicas,
            "puntaje": max(d.get("puntaje", 0) for d in detecciones),
            "propuestas_fusionadas": len(detecciones)
        })

    return fusionados


def refinar_borde_superior(resultados, bordes):
    """Recorta margen superior uniforme hasta encontrar bordes del objeto."""

    refinados = []

    for resultado in resultados:
        refinado = resultado.copy()
        x1, y1, x2, y2 = refinado["caja"]
        alto_caja = y2 - y1
        ancho_caja = x2 - x1

        if alto_caja < 5 or ancho_caja < 5:
            refinados.append(refinado)
            continue

        perfil = np.mean(bordes[y1:y2, x1:x2], axis=1)
        ventana = min(5, len(perfil))
        suavizado = np.array([
            np.mean(perfil[i:i + ventana])
            for i in range(len(perfil) - ventana + 1)
        ])

        limite_busqueda = min(len(suavizado), int(alto_caja * 0.45))
        referencia = np.median(suavizado[:limite_busqueda])
        umbral = max(0.06, referencia * 1.6)
        nuevo_y1 = y1

        for i in range(limite_busqueda - 2):
            if np.mean(suavizado[i:i + 3]) >= umbral:
                nuevo_y1 = y1 + i
                break

        if nuevo_y1 > y1:
            refinado["caja"] = [x1, nuevo_y1, x2, y2]
            caracteristicas = extraer_caracteristicas(refinado["caja"], bordes)
            clase, confianza = clasificar_region(caracteristicas)
            refinado["caracteristicas"] = caracteristicas
            refinado["clase"] = clase
            refinado["confianza"] = confianza
            refinado["recorte_superior_px"] = nuevo_y1 - y1
        else:
            refinado["recorte_superior_px"] = 0

        refinados.append(refinado)

    return refinados


# ============================================
# 14. INTEGRACION DEL DETECTOR YOLO NATIVO 6.1
# ============================================

def detectar_objetos(ruta_imagen, mostrar=True, imprimir=True):
    """Ejecuta el flujo completo y devuelve detecciones y estadisticas."""

    imagen = cargar_imagen(ruta_imagen)
    gris = escala_grises(imagen)
    bordes = detectar_bordes(gris)
    bordes_conectados = dilatar(bordes, 2)
    componentes = componentes_conectados(bordes_conectados)

    alto, ancho = bordes.shape
    cajas = generar_cajas(componentes, alto, ancho)
    distancia_maxima = max(alto, ancho) * 0.06
    cajas_agrupadas = agrupar_cajas(cajas, distancia_maxima)

    propuestas = generar_propuestas(bordes)
    cajas_propuestas = [propuesta["caja"] for propuesta in propuestas]
    resultados = analizar_cajas(cajas_propuestas, bordes)

    for resultado, propuesta in zip(resultados, propuestas):
        resultado["puntaje"] = propuesta["puntaje"]

    umbral_iou = 0.20
    maximo_detecciones = 3
    resultados_nms = supresion_no_maxima(
        resultados,
        umbral_iou,
        maximo_detecciones
    )
    resultados_fusionados = fusionar_cajas_solapadas(resultados_nms, bordes)
    resultados_finales = refinar_borde_superior(resultados_fusionados, bordes)

    estadisticas = {
        "componentes": len(componentes),
        "cajas_iniciales": len(cajas),
        "cajas_agrupadas_referencia": len(cajas_agrupadas),
        "ventanas_propuestas": len(cajas_propuestas),
        "candidatas": sum(r["clase"] == "Objeto candidato" for r in resultados),
        "conservadas_nms": len(resultados_nms),
        "finales": len(resultados_finales),
        "umbral_iou": umbral_iou,
        "maximo_detecciones": maximo_detecciones
    }

    if imprimir:
        print("================================")
        print(" RESULTADOS YOLO NATIVO 6.1")
        print("================================")
        print("Componentes encontrados:", estadisticas["componentes"])
        print("Cajas iniciales:", estadisticas["cajas_iniciales"])
        print("Cajas agrupadas por componentes (referencia):",
              estadisticas["cajas_agrupadas_referencia"])
        print("Ventanas propuestas:", estadisticas["ventanas_propuestas"])
        print("Cajas candidatas antes de NMS:", estadisticas["candidatas"])
        print("Cajas conservadas despues de NMS:", estadisticas["conservadas_nms"])
        print("Cajas finales despues de fusion:", estadisticas["finales"])
        print("Umbral IoU de NMS:", umbral_iou)
        print("Maximo de detecciones mostradas:", maximo_detecciones)
        print("\nRESULTADOS DE CLASIFICACION")

        for i, resultado in enumerate(resultados_finales, start=1):
            print(f"\nRegion {i}")
            print("Clase:", resultado["clase"])
            print("Confianza heuristica:", resultado["confianza"])
            print("Puntaje de ventana:", round(resultado["puntaje"], 3))
            print("Propuestas fusionadas:", resultado["propuestas_fusionadas"])
            print("Recorte superior (pixeles):", resultado["recorte_superior_px"])

            caracteristicas = resultado["caracteristicas"]
            if caracteristicas is not None:
                print("Area:", round(caracteristicas["area"], 4))
                print("Aspecto:", round(caracteristicas["aspecto"], 2))
                print("Densidad:", round(caracteristicas["densidad"], 4))

    if mostrar:
        fig, ejes = plt.subplots(1, 3, figsize=(17, 6))
        ejes[0].imshow(imagen)
        ejes[0].set_title("Imagen original")
        ejes[1].imshow(bordes, cmap="gray")
        ejes[1].set_title("Deteccion Sobel")
        ejes[2].imshow(imagen)

        for resultado in resultados_finales:
            x1, y1, x2, y2 = resultado["caja"]
            color = "lime" if resultado["clase"] == "Objeto candidato" else "red"
            rectangulo = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=color, linewidth=2
            )
            ejes[2].add_patch(rectangulo)
            etiqueta = (
                f"{resultado['clase']}: {resultado['confianza']:.2f} | "
                f"ventana: {resultado['puntaje']:.2f}"
            )
            ejes[2].text(
                x1, max(0, y1 - 8), etiqueta,
                color="white", backgroundcolor=color, fontsize=8
            )

        ejes[2].set_title("Detector YOLO nativo 6.0")
        for ax in ejes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    return {
        "detecciones": resultados_finales,
        "estadisticas": estadisticas
    }


if __name__ == "__main__":
    ruta = sys.argv[1] if len(sys.argv) > 1 else "automovil4.jpeg"
    detectar_objetos(ruta)
