import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
from src.analisis_nlp.BoW.BagOfWords import BagOfWordsAnalyzer, TFIDFAnalyzer

class VisualizadorTSNEBowTfidf:
    """t-SNE para BoW o TF-IDF (ya entrenados)."""

    def __init__(self, corpus=None, labels=None, metodo='bow', max_features=50):
        """
        corpus: lista de letras
        labels: géneros
        metodo: 'bow' o 'tfidf'
        """
        self.corpus = corpus
        self.labels = labels
        self.metodo = metodo.lower()
        self.max_features = max_features

        if self.metodo == 'bow':
            self.vectorizer = BagOfWordsAnalyzer(max_features=self.max_features)
        elif self.metodo == 'tfidf':
            self.vectorizer = TFIDFAnalyzer(max_features=self.max_features)
        else:
            raise ValueError("Método debe ser 'bow' o 'tfidf'")

        self.matrix = None

    def proyectar_datos(self):
        if not self.corpus or not self.labels:
            return None

        # Convertimos a matriz
        self.matrix = self.vectorizer.fit_transform(self.corpus).toarray()

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        datos2d = tsne.fit_transform(self.matrix)

        return {
            'x': datos2d[:, 0].tolist(),
            'y': datos2d[:, 1].tolist(),
            'genero': [g.title() for g in self.labels],
            'info': [c[:60] + '...' for c in self.corpus]  # recortamos la letra para tooltip
        }

    def generar_grafico(self, data):
        if not data:
            return None
        df = pd.DataFrame(data)
        fig = px.scatter(
            df, x='x', y='y', color='genero', hover_name='info',
            title=f"t-SNE {self.metodo.upper()} — Proyección de Géneros Reales",
            template="plotly_white"
        )
        return fig