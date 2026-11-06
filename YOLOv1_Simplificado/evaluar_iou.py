"""Evalua las detecciones de YOLO_nativo.py contra anotaciones manuales."""

import json
import sys
from pathlib import Path

from YOLO_nativo import calcular_iou, detectar_objetos


def emparejar_cajas(predicciones, reales, umbral_iou):
    pares = []

    for indice_pred, prediccion in enumerate(predicciones):
        for indice_real, real in enumerate(reales):
            iou = calcular_iou(prediccion, real)
            if iou >= umbral_iou:
                pares.append((iou, indice_pred, indice_real))

    pares.sort(reverse=True)
    predicciones_usadas = set()
    reales_usadas = set()
    coincidencias = []

    for iou, indice_pred, indice_real in pares:
        if indice_pred in predicciones_usadas or indice_real in reales_usadas:
            continue

        predicciones_usadas.add(indice_pred)
        reales_usadas.add(indice_real)
        coincidencias.append((iou, indice_pred, indice_real))

    verdaderos_positivos = len(coincidencias)
    falsos_positivos = len(predicciones) - verdaderos_positivos
    falsos_negativos = len(reales) - verdaderos_positivos
    return coincidencias, falsos_positivos, falsos_negativos


def main():
    ruta_json = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("anotaciones.json")
    umbral_iou = float(sys.argv[2]) if len(sys.argv) > 2 else 0.50

    if not ruta_json.is_file():
        print(f"No se encontro {ruta_json}. Ejecuta primero anotar_cajas.py.")
        return

    anotaciones = json.loads(ruta_json.read_text(encoding="utf-8"))
    carpeta_imagenes = ruta_json.resolve().parent
    total_tp = total_fp = total_fn = 0
    ious_correctos = []
    tp_un_objeto = fp_un_objeto = fn_un_objeto = 0
    ious_un_objeto = []

    print("================================")
    print(" EVALUACION YOLO NATIVO CON IoU")
    print("================================")
    print("Umbral IoU:", umbral_iou)

    for nombre, cajas_reales in anotaciones.items():
        ruta_imagen = carpeta_imagenes / nombre
        if not ruta_imagen.is_file():
            print(f"\nSe omite {nombre}: no se encontro el archivo de imagen.")
            continue

        salida = detectar_objetos(
            str(ruta_imagen),
            mostrar=False,
            imprimir=False
        )
        predicciones = [
            resultado["caja"]
            for resultado in salida["detecciones"]
            if resultado["clase"] == "Objeto candidato"
        ]

        coincidencias, fp, fn = emparejar_cajas(
            predicciones,
            cajas_reales,
            umbral_iou
        )
        tp = len(coincidencias)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        ious_correctos.extend(iou for iou, _, _ in coincidencias)

        # Este detector se presenta inicialmente para un objeto principal
        # por imagen; el conjunto completo se conserva como diagnostico.
        if len(cajas_reales) == 1:
            tp_un_objeto += tp
            fp_un_objeto += fp
            fn_un_objeto += fn
            ious_un_objeto.extend(iou for iou, _, _ in coincidencias)

        print(f"\n{nombre}")
        print(f"Cajas reales: {len(cajas_reales)} | Predicciones: {len(predicciones)}")
        print(f"TP: {tp} | FP: {fp} | FN: {fn}")

        for iou, indice_pred, indice_real in coincidencias:
            print(
                f"  Coincidencia: prediccion {indice_pred + 1} con caja real "
                f"{indice_real + 1}; IoU={iou:.3f}"
            )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou_promedio = sum(ious_correctos) / len(ious_correctos) if ious_correctos else 0.0

    print("\nRESUMEN DEL CONJUNTO")
    print(f"TP: {total_tp} | FP: {total_fp} | FN: {total_fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"IoU promedio de coincidencias: {iou_promedio:.3f}")

    precision_un_objeto = (
        tp_un_objeto / (tp_un_objeto + fp_un_objeto)
        if tp_un_objeto + fp_un_objeto else 0.0
    )
    recall_un_objeto = (
        tp_un_objeto / (tp_un_objeto + fn_un_objeto)
        if tp_un_objeto + fn_un_objeto else 0.0
    )
    f1_un_objeto = (
        2 * precision_un_objeto * recall_un_objeto /
        (precision_un_objeto + recall_un_objeto)
        if precision_un_objeto + recall_un_objeto else 0.0
    )
    iou_un_objeto = (
        sum(ious_un_objeto) / len(ious_un_objeto)
        if ious_un_objeto else 0.0
    )

    print("\nSUBCONJUNTO: UNA CAJA REAL POR IMAGEN")
    print(f"TP: {tp_un_objeto} | FP: {fp_un_objeto} | FN: {fn_un_objeto}")
    print(f"Precision: {precision_un_objeto:.3f}")
    print(f"Recall: {recall_un_objeto:.3f}")
    print(f"F1: {f1_un_objeto:.3f}")
    print(f"IoU promedio de coincidencias: {iou_un_objeto:.3f}")


if __name__ == "__main__":
    main()
