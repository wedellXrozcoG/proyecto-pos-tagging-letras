#aquí solo va el código clustering bert
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import plotly.express as px

class AnalizadorClustering:
    def __init__(self, col_mongo):
        self.col = col_mongo

    def ejecutar_analisis(self, n_samples=400):
        # devuelve diccionario para el store
        docs = list(self.col.aggregate([
            {"$match": {"embeddings.beto_cls": {"$exists": True}}},
            {"$sample": {"size": n_samples}}
        ]))

        if not docs:
            return None

        vectores = np.array([d["embeddings"]["beto_cls"] for d in docs], dtype='float32')
        titulos = [d.get("titulo", "?") for d in docs]

        # K-Means
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(vectores)
        score = float(silhouette_score(vectores, clusters))

        # t-SNE para poder visualizar en 2D
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
        #lsito para el dash
        df = pd.DataFrame(data['df_2d'])
        fig = px.scatter(
            df, x='x', y='y', color='cluster', hover_name='titulo',
            title=f"K-Means — BERT (Silhouette: {data['score']:.4f})",
            template="plotly_white" # Lo pasamos a blanco para tu nuevo diseño
        )
        return fig