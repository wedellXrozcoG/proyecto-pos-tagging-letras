# ---db_manager.py - Conexión y operaciones con MongoDB---

from pymongo import MongoClient
from datetime import datetime
import pandas as pd
from nltk import word_tokenize, pos_tag
import string
import spacy
nlp = spacy.load("en_core_web_sm")


# ---Conexión de MongoDB---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME   = "dbx_Canciones"
COL_NAME  = "canciones"

# ---Conexión a la base de datos---
def get_collection():
    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]
    return db[COL_NAME]

# ---Agregar canciones---
def insertar_cancion(col, titulo, artista, genero, letra,
                     idioma="ENG", fuente="scraping", url_fuente="",
                     anio=None, pos_tags=None, metricas=None):
    if col.find_one({"titulo": titulo, "artista": artista}):
        return None  # ya existe, no duplicar

    doc = {
        "titulo":             titulo or "Desconocido",
        "artista":            artista or "Desconocido",
        "genero":             genero or "Desconocido",
        "anio":               anio or "Desconocido",
        "letra":              letra or "Desconocido",
        "idioma":             idioma or "Desconocido",
        "fuente":             fuente or "Desconocido",
        "url_fuente":         url_fuente or "Desconocido",
        "fecha_recopilacion": datetime.utcnow(),
        "pos_tags":           pos_tags or {},
        "embeddings":         {},
        "metricas": metricas or { # <-- Usamos las métricas calculadas
            "num_palabras": 0,
            "densidad_lexica": 0.0,
            "ratio_sustantivos_verbos": 0.0
        },
    }
    result = col.insert_one(doc)
    return result.inserted_id

# ---Etiquetas spacy---
def obtener_pos_tags_spacy(texto, nlp_model):

    if not texto or texto == "Desconocido":
        return []

    doc = nlp_model(texto)
    # Lista de etiquetas para esta canción específica
    tags = [[token.text, token.pos_] for token in doc if not token.is_stop]
    return tags

# ---Etiquetas NLTK---
def obtener_pos_tags_nltk(texto):

    if not texto or texto == "Desconocido":
        return []

    # 1. Tokenización
    tokens = word_tokenize(texto)
    # 2. Etiquetado POS
    pos_tags = pos_tag(tokens)
    # 3. Limpieza eliminando puntuación
    puntuacion = set(string.punctuation)
    tags_limpios = [[word, tag] for word, tag in pos_tags if word not in puntuacion]

    return tags_limpios


def calcular_metricas_nlp(tags_spacy):
    if not tags_spacy:
        return {"num_palabras": 0, "densidad_lexica": 0.0, "ratio_sustantivos_verbos": 0.0}

    total_tokens = len(tags_spacy)

    # Definimos qué etiquetas se consideran 'palabras con contenido' (Content Words)
    content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}

    # Contadores
    palabras_contenido = 0
    sustantivos = 0
    verbos = 0

    for token, pos in tags_spacy:
        if pos in content_pos:
            palabras_contenido += 1
        if pos in {'NOUN', 'PROPN'}:
            sustantivos += 1
        if pos == 'VERB':
            verbos += 1

    # Cálculos finales
    densidad = palabras_contenido / total_tokens if total_tokens > 0 else 0
    ratio_sv = sustantivos / verbos if verbos > 0 else sustantivos

    return {
        "num_palabras": total_tokens,
        "densidad_lexica": round(densidad, 2),
        "ratio_sustantivos_verbos": round(ratio_sv, 2)
    }

def migrar_csv_a_mongo(ruta_csv, col, genero_col="Genre", letra_col="text"):

    # Usamos sep=';' porque vimos que tu archivo viene así
    df = pd.read_csv(ruta_csv, sep=';').fillna("")
    insertados = 0

    for i, row in df.iterrows():
        letra_cancion = str(row.get(letra_col, ''))
        # --- Procesamiento NLTK ---
        mis_tags_nltk = obtener_pos_tags_nltk(letra_cancion)

        # --- Procesamiento SPACY ---
        mis_tags_spacy = obtener_pos_tags_spacy(letra_cancion, nlp)

        res_metricas = calcular_metricas_nlp(mis_tags_spacy)

        titulo_val = f"Track_{str(row.get(genero_col))}_{i}" # Al no tener titulo de la cancion se agregro un consecutivo
        artista_val = "Desconocido"  # Un nombre fijo en este caso para el csv
        genero_val = str(row.get(genero_col, "Desconocido"))
        anio_val = str(row.get('Release Date', 'N/A'))

        _id = insertar_cancion(
            col,
            titulo=titulo_val,
            artista=artista_val,
            genero=genero_val,
            letra=letra_cancion,
            anio=anio_val,
            fuente="csv",
            pos_tags={
                "nltk": mis_tags_nltk,
                "spacy": mis_tags_spacy
            },
            metricas=res_metricas
        )
        if _id:
            insertados += 1

    print(f"Migradas {insertados} canciones nuevas a MongoDB.")
    return