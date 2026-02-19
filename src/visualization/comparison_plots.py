import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


class POSVisualizer:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, sep=';')

    def comparar_distribucion(self):
        # Contar tags
        nltk_counts = Counter(self.df['nltk_tag'])
        spacy_counts = Counter(self.df['spacy_pos'])

        # Top 10 de cada uno
        nltk_top10 = nltk_counts.most_common(10)
        spacy_top10 = spacy_counts.most_common(10)

        # Crear subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # NLTK
        nltk_names = [t[0] for t in nltk_top10]
        nltk_values = [t[1] for t in nltk_top10]
        ax1.bar(nltk_names, nltk_values, color=plt.cm.Set3(range(10)), edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('NLTK Tag', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
        ax1.set_title('Top 10 NLTK Tags', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # Spacy
        spacy_names = [t[0] for t in spacy_top10]
        spacy_values = [t[1] for t in spacy_top10]
        ax2.bar(spacy_names, spacy_values, color=plt.cm.Set2(range(10)), edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Spacy POS', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
        ax2.set_title('Top 10 Spacy POS', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.show()

    def mostrar_metricas(self):
        """Muestra métricas comparativas"""
        nltk_total = len(self.df['nltk_tag'])
        spacy_total = self.df['spacy_pos'].notna().sum()

        print("📊 MÉTRICAS COMPARATIVAS")
        print("=" * 50)
        print(f"Total tokens NLTK:  {nltk_total:,}")
        print(f"Total tokens Spacy: {spacy_total:,}")
        print(f"\nTipos únicos NLTK:  {self.df['nltk_tag'].nunique()}")
        print(f"Tipos únicos Spacy: {self.df['spacy_pos'].nunique()}")


# Uso
visualizer = POSVisualizer("../../data/results/comparisons.csv")
visualizer.comparar_distribucion()
visualizer.mostrar_metricas()