import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import plotly.express as px
from src.analisis_nlp.BoW.BagOfWords import BagOfWordsAnalyzer, TFIDFAnalyzer, CorpusLoader

class AnalizadorClusteringBowTfidf:
    """Clustering de canciones usando BoW o TF-IDF, listo para Dash."""

    def __init__(self, col_mongo, max_features: int = 50):
        self.col = col_mongo
        self.max_features = max_features
        self.bow_analyzer = BagOfWordsAnalyzer(max_features=max_features)
        self.tfidf_analyzer = TFIDFAnalyzer(max_features=max_features)

    def ejecutar_analisis(self, metodo='bow', n_samples=400):
        """Devuelve diccionario con clusters + t-SNE listo para store."""
        # Carga datos desde Mongo
        loader = CorpusLoader(sample_size=n_samples)
        corpus, titulos = loader.load(self.col)
        if not corpus:
            return None

        # Representación
        if metodo.lower() == 'bow':
            X = self.bow_analyzer.fit_transform(corpus).toarray()
        elif metodo.lower() == 'tfidf':
            X = self.tfidf_analyzer.fit_transform(corpus).toarray()
        else:
            raise ValueError("Método debe ser 'bow' o 'tfidf'")

        # K-Means
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        score = float(silhouette_score(X, clusters))

        # t-SNE 2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        datos2d = tsne.fit_transform(X)

        return {
            'score': score,
            'df_2d': {
                'x': datos2d[:, 0].tolist(),
                'y': datos2d[:, 1].tolist(),
                'cluster': [f"Cluster {c}" for c in clusters],
                'titulo': titulos
            }
        }

    def generar_grafico(self, data, metodo='BoW'):
        """Devuelve gráfico Plotly listo para Dash."""
        df = pd.DataFrame(data['df_2d'])
        fig = px.scatter(
            df, x='x', y='y', color='cluster', hover_name='titulo',
            title=f"K-Means — {metodo} (Silhouette: {data['score']:.4f})",
            template="plotly_white"
        )
        return fig