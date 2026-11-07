# ============================================
# YOLO NATIVO 7.2
# Detector nativo de automoviles: HOG, clasificador lineal, IoU y NMS
# ============================================

import sys
from pathlib import Path
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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
# 3. IoU Y SUPRESION DE CAJAS DUPLICADAS (NMS)
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


def supresion_no_maxima(resultados, umbral_iou=0.20, maximo=6,
                         puntaje_minimo=0.50):
    """Quita cajas repetidas por IoU o por cercania de sus centros."""

    candidatos = [
        resultado for resultado in resultados
        if resultado["clase"] == "Objeto candidato"
        and resultado.get("puntaje", resultado["confianza"]) >= puntaje_minimo
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

        x1, y1, x2, y2 = mejor["caja"]
        ancho_a, alto_a = x2 - x1, y2 - y1
        centro_ax, centro_ay = (x1 + x2) / 2, (y1 + y2) / 2
        restantes = []
        for resultado in candidatos:
            bx1, by1, bx2, by2 = resultado["caja"]
            ancho_b, alto_b = bx2 - bx1, by2 - by1
            centro_bx, centro_by = (bx1 + bx2) / 2, (by1 + by2) / 2
            centros_cercanos = (
                abs(centro_ax - centro_bx) <= 0.65 * min(ancho_a, ancho_b)
                and abs(centro_ay - centro_by) <= 0.65 * min(alto_a, alto_b)
            )
            se_solapan = calcular_iou(mejor["caja"], resultado["caja"]) > umbral_iou
            if not centros_cercanos and not se_solapan:
                restantes.append(resultado)
        candidatos = restantes

    return conservados


# ============================================
# 4. DESCRIPTOR HOG Y ENTRENAMIENTO CON PASCAL VOC 2007
# ============================================

def descriptor_hog(region):
    """Descriptor HOG sencillo, calculado con gradientes y histogramas propios."""

    recorte = np.asarray(
        Image.fromarray(np.uint8(np.clip(region, 0, 255))).resize((64, 32))
    ).astype(float)
    gy, gx = np.gradient(recorte)
    magnitud = np.sqrt(gx * gx + gy * gy)
    orientacion = (np.degrees(np.arctan2(gy, gx)) + 180) % 180
    bins = (orientacion // 20).astype(int) % 9

    descriptor = []
    for y in range(0, 32, 8):
        for x in range(0, 64, 8):
            hist = np.bincount(
                bins[y:y + 8, x:x + 8].ravel(),
                weights=magnitud[y:y + 8, x:x + 8].ravel(),
                minlength=9
            )
            descriptor.extend(hist)
    descriptor = np.asarray(descriptor, dtype=float)
    descriptor /= np.linalg.norm(descriptor) + 1e-8
    return descriptor


def leer_anotaciones_voc(ruta_xml):
    raiz = ElementTree.parse(ruta_xml).getroot()
    objetos = []
    for objeto in raiz.findall("object"):
        nombre = objeto.findtext("name", default="").strip().lower()
        dificil = objeto.findtext("difficult", default="0") == "1"
        caja = objeto.find("bndbox")
        if caja is None:
            continue
        # VOC usa coordenadas 1-indexadas; se convierten a caja [x1,y1,x2,y2).
        x1 = int(float(caja.findtext("xmin", "1"))) - 1
        y1 = int(float(caja.findtext("ymin", "1"))) - 1
        x2 = int(float(caja.findtext("xmax", "1")))
        y2 = int(float(caja.findtext("ymax", "1")))
        objetos.append((nombre, [x1, y1, x2, y2], dificil))
    return objetos


def entrenar_modelo_voc(ruta_dataset=None):
    """Entrena un clasificador lineal propio con coches y fondos de VOC train."""

    base = Path(ruta_dataset) if ruta_dataset else (
        Path(__file__).parent / "datasets" / "VOC2007" / "VOCdevkit" / "VOC2007"
    )
    ids_file = base / "ImageSets" / "Main" / "train.txt"
    if not ids_file.is_file():
        raise FileNotFoundError(
            f"No encuentro {ids_file}. Descomprime VOC2007 trainval en datasets/VOC2007."
        )

    rng = np.random.default_rng(7)
    ids = ids_file.read_text(encoding="utf-8").split()
    rng.shuffle(ids)
    ids = ids[:1800]
    positivos, negativos, aspectos, escalas = [], [], [], []

    for indice, identificador in enumerate(ids):
        ruta_imagen = base / "JPEGImages" / f"{identificador}.jpg"
        ruta_xml = base / "Annotations" / f"{identificador}.xml"
        if not ruta_imagen.is_file() or not ruta_xml.is_file():
            continue

        imagen = escala_grises(cargar_imagen(str(ruta_imagen)))
        alto, ancho = imagen.shape
        objetos = leer_anotaciones_voc(ruta_xml)
        cajas_auto = [c for nombre, c, dificil in objetos if nombre == "car" and not dificil]
        cajas_objeto = [c for _, c, _ in objetos]

        for caja in cajas_auto:
            x1, y1, x2, y2 = caja
            if x2 <= x1 or y2 <= y1:
                continue
            recorte = imagen[max(0, y1):min(alto, y2), max(0, x1):min(ancho, x2)]
            if recorte.size == 0:
                continue
            positivos.append(descriptor_hog(recorte))
            positivos.append(descriptor_hog(recorte[:, ::-1]))
            aspectos.append((x2 - x1) / max(1, y2 - y1))
            escalas.append((x2 - x1) / max(1, ancho))

        # Dos ventanas de fondo por imagen, sin solapar objetos anotados.
        aspecto_base = float(np.median(aspectos)) if aspectos else 1.8
        aceptados = 0
        intentos = 0
        while aceptados < 2 and intentos < 20:
            intentos += 1
            proporcion = float(rng.choice([0.16, 0.22, 0.30, 0.40, 0.52, 0.64]))
            ww = max(24, int(ancho * proporcion))
            wh = max(16, int(ww / aspecto_base))
            if ww >= ancho or wh >= alto:
                continue
            x1 = int(rng.integers(0, ancho - ww + 1))
            y1 = int(rng.integers(0, alto - wh + 1))
            caja_fondo = [x1, y1, x1 + ww, y1 + wh]
            if any(calcular_iou(caja_fondo, caja) > 0.05 for caja in cajas_objeto):
                continue
            negativos.append(descriptor_hog(imagen[y1:y1 + wh, x1:x1 + ww]))
            aceptados += 1

        if (indice + 1) % 300 == 0:
            print(f"VOC train procesadas: {indice + 1}/{len(ids)}")

    if len(positivos) < 100 or len(negativos) < 100:
        raise ValueError("VOC no produjo suficientes ejemplos positivos y de fondo.")

    datos = np.vstack((positivos, negativos))
    etiquetas = np.r_[np.ones(len(positivos)), np.zeros(len(negativos))]
    pesos_muestra = np.where(
        etiquetas == 1, 0.5 / len(positivos), 0.5 / len(negativos)
    )
    pesos = np.zeros(datos.shape[1], dtype=float)
    sesgo = 0.0
    for _ in range(500):
        p = 1 / (1 + np.exp(-np.clip(datos @ pesos + sesgo, -20, 20)))
        error = (p - etiquetas) * pesos_muestra
        pesos -= 1.0 * (datos.T @ error + 0.001 * pesos)
        sesgo -= np.sum(error)

    destino = Path(__file__).with_name("modelo_autos_voc.npz")
    np.savez_compressed(
        destino,
        pesos=pesos,
        sesgo=sesgo,
        aspectos=np.percentile(aspectos, [20, 50, 80]),
        escalas=np.percentile(escalas, [10, 30, 50, 70, 90])
    )
    print("Modelo nativo entrenado y guardado en:", destino)
    print("Recortes positivos:", len(positivos), "| fondos:", len(negativos))


def cargar_modelo_voc():
    ruta = Path(__file__).with_name("modelo_autos_voc.npz")
    if not ruta.is_file():
        raise FileNotFoundError(
            "Primero ejecuta: py -3 YOLO_nativo.py --entrenar-voc"
        )
    with np.load(ruta) as datos:
        return {nombre: datos[nombre] for nombre in datos.files}


def proponer_automoviles(gris, modelo, umbral=0.50):
    alto, ancho = gris.shape
    escalas = sorted(set(np.clip(
        np.r_[np.linspace(0.14, 0.82, 9), modelo["escalas"]], 0.12, 0.90
    ).round(2)))
    candidatos = []
    aspectos = modelo["aspectos"]

    for aspecto in aspectos:
        for escala in escalas:
            ww = int(ancho * escala)
            wh = int(ww / aspecto)
            if ww < 24 or wh < 16 or ww > ancho or wh > alto:
                continue
            paso = max(10, ww // 5)
            for y1 in range(0, alto - wh + 1, paso):
                for x1 in range(0, ancho - ww + 1, paso):
                    caja = [x1, y1, x1 + ww, y1 + wh]
                    descriptor = descriptor_hog(gris[y1:y1 + wh, x1:x1 + ww])
                    logit = float(descriptor @ modelo["pesos"] + modelo["sesgo"])
                    confianza = 1 / (1 + np.exp(-np.clip(logit, -20, 20)))
                    if confianza >= umbral:
                        candidatos.append({
                            "caja": caja,
                            "clase": "Objeto candidato",
                            "confianza": confianza,
                            "puntaje": confianza
                        })
    return candidatos


# ============================================
# 5. DETECCION Y VISUALIZACION
# ============================================

def detectar_objetos(ruta_imagen, mostrar=True, imprimir=True):
    """Detecta con el clasificador propio entrenado desde VOC2007."""

    imagen = cargar_imagen(ruta_imagen)
    gris = escala_grises(imagen)
    modelo = cargar_modelo_voc()
    confianza_minima = 0.50
    umbral_iou = 0.20
    maximo_detecciones = 6
    candidatos = proponer_automoviles(gris, modelo, confianza_minima)
    resultados_finales = supresion_no_maxima(
        candidatos, umbral_iou, maximo=maximo_detecciones,
        puntaje_minimo=confianza_minima
    )
    estadisticas = {
        "ventanas_candidatas": len(candidatos),
        "detecciones": len(resultados_finales),
        "confianza_minima": confianza_minima,
        "umbral_iou": umbral_iou
    }

    if imprimir:
        print("================================")
        print(" DETECTOR NATIVO DE AUTOMOVILES 7.2")
        print("================================")
        print("Ventanas candidatas:", len(candidatos))
        print("Automoviles despues de NMS:", len(resultados_finales))
        print("Confianza minima:", confianza_minima)
        print("Umbral IoU:", umbral_iou)
        print("Maximo de detecciones:", maximo_detecciones)
        for i, resultado in enumerate(resultados_finales, start=1):
            print(f"\nAutomovil {i}")
            print("Confianza del clasificador:", round(resultado["confianza"], 3))
            print("Caja:", resultado["caja"])

    if mostrar:
        fig, ejes = plt.subplots(1, 2, figsize=(12, 5))
        ejes[0].imshow(imagen)
        ejes[0].set_title("Imagen original")
        ejes[1].imshow(imagen)
        for resultado in resultados_finales:
            x1, y1, x2, y2 = resultado["caja"]
            rectangulo = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor="lime", linewidth=2
            )
            ejes[1].add_patch(rectangulo)
            ejes[1].text(
                x1, max(0, y1 - 8),
                f"Automovil: {resultado['confianza']:.2f}",
                color="white", backgroundcolor="green", fontsize=8
            )
        ejes[1].set_title("Detector nativo entrenado con VOC 7.2")
        for ax in ejes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    return {"detecciones": resultados_finales, "estadisticas": estadisticas}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--entrenar-voc":
        raiz_dataset = sys.argv[2] if len(sys.argv) > 2 else None
        entrenar_modelo_voc(raiz_dataset)
    else:
        ruta = sys.argv[1] if len(sys.argv) > 1 else "automovil.jpeg"
        detectar_objetos(ruta)
