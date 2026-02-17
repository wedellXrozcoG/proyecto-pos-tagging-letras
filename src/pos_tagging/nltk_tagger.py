import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


class nltkPosTagger:
    def __init__(self, filepath):
        self.corpus = pd.read_csv(filepath, sep=';')
        self.all_pos_tags = []
        self.pos_counts = None
        self.puntuacion = {'.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '-', '--', '...', '`', '``', "''"}

    def ejemplo_una_cancion(self):
        """Muestra ejemplo con la primera canción"""
        sentence = self.corpus['text'].iloc[0]

        print("ORIGINAL:")
        print(f'"{sentence}"\n')

        tokens = word_tokenize(sentence) # Paso1: aquí se tokeniza la oración
        print("TOKENS:")
        print(tokens)
        print()

        pos_tags_nltk = pos_tag(tokens) # Paso2: se relaizan los postagging con NLTK
        print("POS TAGGING CON NLTK:")
        for word, tag in pos_tags_nltk:
            print(f"{word:15} → {tag}")

    def analizar_multiples_canciones(self, n_ejemplos=5):
        """Analiza múltiples canciones (muestra primeras n)"""
        print("\n" + "=" * 60)
        print("ANALIZANDO MÚLTIPLES CANCIONES CON NLTK")
        print("=" * 60 + "\n")

        n_canciones = len(self.corpus)

        for idx in range(n_canciones): #para que la salida sea solo un par de canciones
            sentence = self.corpus['text'].iloc[idx] #canción
            emotion = self.corpus['emotion'].iloc[idx] #para que salga la emoción de la canción tokenizada
            genre = self.corpus['Genre'].iloc[idx] #para que pueda aparecer el género de la canción tokenizada

            print(f"Canción {idx + 1} - Emoción: {emotion}")
            print(f"Genero: {genre}")
            print(f"Texto: {sentence[:100]}...")

            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)

            for word, tag in pos_tags:
                print(f"  {word:15} → {tag}")
            print()

            if idx == n_ejemplos - 1:
                print(f"\n... (procesando las {n_canciones - n_ejemplos} canciones restantes en segundo plano)")
                break

        print(f"\n✓ Total procesadas: {n_canciones} canciones")

    def procesar_corpus_completo(self): #procesa las canciones (sin imprimirlas porque son 10 mil)
        print("Procesando canciones...")

        for idx in range(len(self.corpus)):
            tokens = word_tokenize(self.corpus['text'].iloc[idx])
            pos_tags = pos_tag(tokens)
            self.all_pos_tags.extend([tag for word, tag in pos_tags if word not in self.puntuacion])

        self.pos_counts = Counter(self.all_pos_tags)
        print(f"✓ {len(self.corpus)} canciones procesadas\n")

    def visualizar_top10(self): # gráfico con el top 10 de los pos tags del corpus
        top_10 = self.pos_counts.most_common(10)
        names = [t[0] for t in top_10]
        values = [t[1] for t in top_10]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, values, color=plt.cm.Set3(range(10)), edgecolor='black', linewidth=1.5)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height):,}',
                    ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('POS Tag', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
        ax.set_title('Top 10 POS Tags - Hip-Hop - Pop ', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

    def mostrar_metricas(self):
        """Muestra métricas y análisis morfológico"""
        top_10 = self.pos_counts.most_common(10)

        print(f"📊 Total palabras: {len(self.all_pos_tags):,}")
        print(f"📊 Tipos diferentes de POS tags: {len(self.pos_counts)}\n")

        print("🏆 TOP 10 POS tags más frecuentes:")
        for tag, count in top_10:
            percentage = (count / len(self.all_pos_tags)) * 100
            print(f"  {tag:10} → {count:,} ({percentage:.2f}%)")

        # Análisis por categoría gramatical
        sustantivos = sum([count for tag, count in self.pos_counts.items() if tag.startswith('NN')])
        verbos = sum([count for tag, count in self.pos_counts.items() if tag.startswith('VB')])
        adjetivos = sum([count for tag, count in self.pos_counts.items() if tag.startswith('JJ')])
        adverbios = sum([count for tag, count in self.pos_counts.items() if tag.startswith('RB')])

        print(f"\n📝 Distribución por categoría:")
        print(f"  Sustantivos: {sustantivos:,} ({sustantivos / len(self.all_pos_tags) * 100:.2f}%)")
        print(f"  Verbos:      {verbos:,} ({verbos / len(self.all_pos_tags) * 100:.2f}%)")
        print(f"  Adjetivos:   {adjetivos:,} ({adjetivos / len(self.all_pos_tags) * 100:.2f}%)")
        print(f"  Adverbios:   {adverbios:,} ({adverbios / len(self.all_pos_tags) * 100:.2f}%)")
