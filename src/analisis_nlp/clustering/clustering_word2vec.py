import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import plotly.express as px


class AnalizadorClusteringWord2Vec:
    """Aplica K-Means y t-SNE sobre embeddings Word2Vec promedio guardados en MongoDB."""

    def __init__(self, col_mongo):
        self.col = col_mongo

    def ejecutar_analisis(self, n_samples=400):
        # CAMBIO 1: Buscamos documentos que tengan el array word2vec_avg
        docs = list(self.col.aggregate([
            {"$match": {"embeddings.word2vec_avg": {"$exists": True}}},
            {"$sample": {"size": n_samples}}
        ]))

        if not docs:
            return None

        # CAMBIO 2: Extraemos los vectores de 100 dimensiones de Word2Vec
        vectores = np.array([d["embeddings"]["word2vec_avg"] for d in docs], dtype='float32')
        titulos = [d.get("titulo", "?") for d in docs]

        # K-Means (Agrupación no supervisada en 3 géneros esperados)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(vectores)

        # Evaluamos qué tan bien se separaron los grupos
        score = float(silhouette_score(vectores, clusters))

        # t-SNE para poder visualizar las 100 dimensiones en un plano 2D (Requisito de la rúbrica)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        datos2d = tsne.fit_transform(vectores)

        return {
            'score': score,
            'df_2d': {
                'x': datos2d[:, 0].tolist(),
                'y': datos2d[:, 1].tolist(),
                'cluster': [f"Cluster {c}" for c in clusters],
                'titulo': titulos
            }
        }

    def generar_grafico(self, data):
        # Listo para el dash
        df = pd.DataFrame(data['df_2d'])

        # CAMBIO 3: Actualizamos el título para identificar que es el modelo Word2Vec
        fig = px.scatter(
            df, x='x', y='y', color='cluster', hover_name='titulo',
            title=f"K-Means — Word2Vec (Silhouette: {data['score']:.4f})",
            template="plotly_white"
        )
        return fig