"""Anota cajas reales para evaluar YOLO_nativo.py con IoU."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image


TAMANO_IMAGEN = (600, 600)
ARCHIVO_SALIDA = Path("anotaciones.json")


def cargar_rutas():
    if len(sys.argv) > 1:
        rutas = [Path(ruta) for ruta in sys.argv[1:]]
    else:
        rutas = sorted(Path.cwd().glob("automovil*.jpeg"))
        rutas += sorted(Path.cwd().glob("automovil*.jfif"))

    unicas = []
    hashes = set()

    for ruta in rutas:
        if not ruta.is_file():
            print(f"No se encontro: {ruta}")
            continue

        huella = hashlib.sha256(ruta.read_bytes()).hexdigest()
        if huella not in hashes:
            hashes.add(huella)
            unicas.append(ruta.resolve())

    return unicas


def guardar_anotaciones(anotaciones):
    ARCHIVO_SALIDA.write_text(
        json.dumps(anotaciones, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def anotar_imagen(ruta, cajas_existentes):
    imagen = Image.open(ruta).convert("RGB")
    imagen.thumbnail(TAMANO_IMAGEN)

    figura, eje = plt.subplots(figsize=(10, 8))
    eje.imshow(imagen)
    eje.set_title(
        f"{ruta.name}\nDibuja una caja por auto | N: siguiente | U: deshacer | Q: salir"
    )
    eje.axis("off")

    cajas = [list(caja) for caja in cajas_existentes]
    rectangulos = []

    def dibujar_caja(caja):
        x1, y1, x2, y2 = caja
        rectangulo = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="lime", linewidth=2
        )
        eje.add_patch(rectangulo)
        rectangulos.append(rectangulo)

    for caja in cajas:
        dibujar_caja(caja)

    def al_dibujar(inicio, fin):
        if inicio.xdata is None or fin.xdata is None:
            return

        ancho, alto = imagen.size
        x1 = max(0, min(ancho - 1, round(min(inicio.xdata, fin.xdata))))
        y1 = max(0, min(alto - 1, round(min(inicio.ydata, fin.ydata))))
        x2 = max(0, min(ancho, round(max(inicio.xdata, fin.xdata))))
        y2 = max(0, min(alto, round(max(inicio.ydata, fin.ydata))))

        if x2 - x1 < 3 or y2 - y1 < 3:
            return

        caja = [x1, y1, x2, y2]
        cajas.append(caja)
        dibujar_caja(caja)
        figura.canvas.draw_idle()

    selector = RectangleSelector(
        eje,
        al_dibujar,
        useblit=True,
        button=[1],
        minspanx=3,
        minspany=3,
        spancoords="data",
        interactive=False,
        props={"edgecolor": "lime", "facecolor": "none", "linewidth": 2}
    )

    estado = {"salir": False}

    def al_teclear(evento):
        if evento.key == "u" and cajas:
            cajas.pop()
            rectangulos.pop().remove()
            figura.canvas.draw_idle()
        elif evento.key == "n":
            plt.close(figura)
        elif evento.key == "q":
            estado["salir"] = True
            plt.close(figura)

    figura.canvas.mpl_connect("key_press_event", al_teclear)
    # Mantener una referencia evita que el selector desaparezca antes de dibujar.
    figura._selector_cajas = selector
    plt.show()

    return cajas, estado["salir"]


def main():
    rutas = cargar_rutas()
    if not rutas:
        print("No encontre imagenes. Pasa sus rutas como argumentos.")
        return

    anotaciones = {}
    if ARCHIVO_SALIDA.exists():
        anotaciones = json.loads(ARCHIVO_SALIDA.read_text(encoding="utf-8"))

    print("Coordenadas en imagen redimensionada a un maximo de 600 x 600 px.")
    print("Dibuja todas las cajas reales de cada imagen; guarda cada imagen con N.")

    for ruta in rutas:
        cajas, salir = anotar_imagen(ruta, anotaciones.get(ruta.name, []))
        anotaciones[ruta.name] = cajas
        guardar_anotaciones(anotaciones)
        print(f"{ruta.name}: {len(cajas)} cajas guardadas en {ARCHIVO_SALIDA}")
        if salir:
            break


if __name__ == "__main__":
    main()
