# PREGUNTA 1: ¿QUÉ PALABRAS SE USARON MÁS POR DÉCADA?

import pandas as pd
import plotly.express as px
import spacy
from collections import Counter


class TemporalAnalyzer:
    def __init__(self, dataframe):
        self.nlp = spacy.load("en_core_web_sm")
        self.corpus = dataframe
        self.puntuacion = {'.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '-', '--', '...', '`', '``', "''"}
        # se hace una caché para qu ecada vez que cambio de tag en la pág principal no dure un montón por la re-ejecución del méto do.
        print("Procesando datos temporales (solo una vez)...")
        self._df_palabras = None  # Cache
        self._df_sentimientos = None  # Cache

    def palabras_mas_usadas_por_decada(self):
        # ✅ Si ya está procesado, retornar directamente
        if self._df_palabras is not None:
            print("Usando cache de palabras...")
            return self._crear_grafico_palabras()

        decadas = {
            '1990s': range(1990, 2000),
            '2000s': range(2000, 2010),
            '2010s': range(2010, 2020),
            '2020s': range(2020, 2025)
        }

        print("TOP 10 PALABRAS MÁS USADAS POR DÉCADA\n")
        resultados = []

        for decada, años in decadas.items():
            canciones_decada = self.corpus[self.corpus['Release Date'].isin(años)]
            todas_palabras = []

            for texto in canciones_decada['text']:
                doc = self.nlp(texto.lower())
                todas_palabras.extend([token.text for token in doc
                                       if not token.is_punct and not token.is_space and token.text.isalnum()])

            palabras_counter = Counter(todas_palabras)

            print(f"📅 {decada}:")
            for palabra, count in palabras_counter.most_common(10):
                print(f"  {palabra:15} → {count:,}")
                resultados.append({'decada': decada, 'palabra': palabra, 'count': count})
            print()

        self._df_palabras = pd.DataFrame(resultados)  # ✅ Guardar cache
        return self._crear_grafico_palabras()

    def _crear_grafico_palabras(self):
        #Gráfico con los datos en caché
        fig = px.bar(self._df_palabras, x='palabra', y='count', color='decada',
                     barmode='group',
                     title='Top 10 Palabras Más Usadas por Década')
        fig.update_layout(xaxis_title='Palabra', yaxis_title='Frecuencia', height=500)
        return fig

    def sentimientos_por_decada(self):
        # esto inidca que si ya está procesado, retorne la cache
        if self._df_sentimientos is not None:
            return self._df_sentimientos

        decadas = {
            '1990s': range(1990, 2000),
            '2000s': range(2000, 2010),
            '2010s': range(2010, 2020),
            '2020s': range(2020, 2025)
        }

        print("SENTIMIENTOS PREDOMINANTES POR DÉCADA\n")
        resultados = []

        for decada, años in decadas.items():
            canciones_decada = self.corpus[self.corpus['Release Date'].isin(años)]
            emociones = Counter(canciones_decada['emotion'])

            print(f"📅 {decada}:")
            for emocion, count in emociones.most_common():
                porcentaje = (count / len(canciones_decada)) * 100
                print(f"  {emocion:15} → {count:,} ({porcentaje:.1f}%)")
                resultados.append({'decada': decada, 'emocion': emocion, 'count': count, 'porcentaje': porcentaje})
            print()

        self._df_sentimientos = pd.DataFrame(resultados)  # SE GUARDA LA CACHÉ
        return self._df_sentimientos

    def visualizar_sentimientos(self):
        df_sentimientos = self.sentimientos_por_decada()  # Usa cache

        fig = px.bar(df_sentimientos, x='decada', y='porcentaje',
                     color='emocion', barmode='group',
                     title='Evolución de Sentimientos en Hip-Hop por Década',
                     labels={'decada': 'Década', 'porcentaje': 'Porcentaje (%)', 'emocion': 'Emoción'})

        fig.update_layout(xaxis_title='Década', yaxis_title='Porcentaje (%)', height=500)
        return fig