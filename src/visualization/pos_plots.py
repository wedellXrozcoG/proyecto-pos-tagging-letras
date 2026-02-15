import sys
import os

# Definir las rutas de las carpetas donde están tus archivos .py
ruta_spacy = r"D:\proyecto-pos-tagging-letras\src\pos_tagging"
ruta_EDA = r"D:\proyecto-pos-tagging-letras\src\data"

# Agregarlas al sistema
for r in [ruta_spacy, ruta_EDA]:
    if r not in sys.path:
        sys.path.append(r)

from spacy_tagger import pos_spacy
from preprocessorEDA import preprocesador

class graficos(pos_spacy, preprocesador):
    def __init__(self, df):
        # Inicializamos ambas clases padre manualmente pasándoles el DataFrame
        pos_spacy.__init__(self, df)
        preprocesador.__init__(self, df)

