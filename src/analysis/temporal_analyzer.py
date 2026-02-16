# PREGUNTA 1: ¿QUÉ SENTIMIENTOS PREDOMINARON POR DÉCADA DESDE LOS 90s HASTA 2020s?

import pandas as pd
import matplotlib.pyplot as plt
from nltk import pos_tag, word_tokenize
from collections import Counter


class TemporalAnalyzer:
    def __init__(self, filepath):
        self.corpus = pd.read_csv(filepath, sep=';')
        self.puntuacion = {'.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '-', '--', '...', '`', '``', "''"}

    def palabras_mas_usadas_por_decada(self):
        """Top palabras más usadas por década"""
        decadas = {
            '1990s': range(1990, 2000),
            '2000s': range(2000, 2010),
            '2010s': range(2010, 2020),
            '2020s': range(2020, 2025)
        }

        print("🔥 TOP 10 PALABRAS MÁS USADAS POR DÉCADA\n")

        for decada, años in decadas.items():
            canciones_decada = self.corpus[self.corpus['Release Date'].isin(años)]
            todas_palabras = []

            for texto in canciones_decada['text']:
                tokens = word_tokenize(texto.lower())
                todas_palabras.extend([t for t in tokens if t not in self.puntuacion and t.isalnum()])

            palabras_counter = Counter(todas_palabras)

            print(f"📅 {decada}:")
            for palabra, count in palabras_counter.most_common(10):
                print(f"  {palabra:15} → {count:,}")
            print()

    def sentimientos_por_decada(self):
        """Analiza qué sentimientos predominaron por década"""
        decadas = {
            '1990s': range(1990, 2000),
            '2000s': range(2000, 2010),
            '2010s': range(2010, 2020),
            '2020s': range(2020, 2025)
        }

        print("😊 SENTIMIENTOS PREDOMINANTES POR DÉCADA\n")

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

        return pd.DataFrame(resultados)

    def visualizar_sentimientos(self):
        """Gráfico de sentimientos por década"""
        df_sentimientos = self.sentimientos_por_decada()

        # Gráfico de barras agrupadas
        decadas = df_sentimientos['decada'].unique()
        emociones = df_sentimientos['emocion'].unique()

        fig, ax = plt.subplots(figsize=(12, 6))

        x = range(len(decadas))
        width = 0.8 / len(emociones)

        for i, emocion in enumerate(emociones):
            datos = df_sentimientos[df_sentimientos['emocion'] == emocion]
            valores = [
                datos[datos['decada'] == d]['porcentaje'].values[0] if len(datos[datos['decada'] == d]) > 0 else 0 for d
                in decadas]
            ax.bar([pos + width * i for pos in x], valores, width, label=emocion)

        ax.set_xlabel('Década', fontsize=12, fontweight='bold')
        ax.set_ylabel('Porcentaje (%)', fontsize=12, fontweight='bold')
        ax.set_title('Evolución de Sentimientos en Hip-Hop por Década', fontsize=14, fontweight='bold')
        ax.set_xticks([pos + width * len(emociones) / 2 for pos in x])
        ax.set_xticklabels(decadas)
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()
