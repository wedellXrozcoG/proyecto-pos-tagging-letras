# Limpieza de letras desde MongoDB para las nuevas letras dedsde scraping
import re
from src.db_manager import get_collection


def limpiar_letra(texto):
    # limpia la letra de la canción quitando lo repetitivo

    if not texto or texto == "Desconocido":
        return texto

    # 1. Eliminar corchetes y su contenido [Verse 1], [Chorus], etc.
    texto = re.sub(r'\[.*?\]', '', texto)

    # 2. Eliminar paréntesis (coros de fondo, anotaciones)
    texto = re.sub(r'\(.*?\)', '', texto)

    # 3. Eliminar sílabas repetitivas sin significado: "la la la", "oh oh oh",
    #    "na na na", "yeah yeah", "hey hey", etc.
    texto = re.sub(r'\b(la|oh|na|yeah|hey|woah|ooh|ah|uh|mm|hmm)(\s+\1){2,}\b',
                   '', texto, flags=re.IGNORECASE)

    # 4. Eliminar líneas que sean solo sílabas repetidas (ej: "La, la, la, la, oh")
    lineas = texto.split(' ')
    lineas_limpias = []
    for linea in lineas:
        palabras = re.findall(r'\b\w+\b', linea.lower())
        unicas   = set(palabras)
        # Si la línea tiene más de 3 palabras y más del 70% son iguales → basura
        if len(palabras) > 3 and len(unicas) <= max(1, len(palabras) * 0.3):
            continue
        lineas_limpias.append(linea)
    texto = ' '.join(lineas_limpias)

    # 5. Eliminar caracteres raros — dejar letras, números, espacios y apóstrofes
    texto = re.sub(r"[^\w\s']", ' ', texto)

    # 6. Colapsar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto


def limpiar_scraping_en_mongo(): # se limpia trayendo los datos desde mongo para luego actualizarlos ahí mismo
    col         = get_collection()
    canciones   = list(col.find({"fuente": "scraping"}))
    actualizadas = 0

    print(f"Canciones de scraping encontradas: {len(canciones)}")

    for cancion in canciones:
        letra_original = cancion.get("letra", "")
        letra_limpia   = limpiar_letra(letra_original)

        # Solo actualizar si hubo cambios
        if letra_limpia != letra_original:
            col.update_one(
                {"_id": cancion["_id"]},
                {"$set": {"letra": letra_limpia}}
            )
            actualizadas += 1

    print(f"Letras actualizadas en MongoDB: {actualizadas}")
    return actualizadas