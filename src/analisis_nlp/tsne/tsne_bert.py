#aquí solo va el código tsne bert
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px

class VisualizadorTSNE:
    def __init__(self, col_mongo):
        self.col = col_mongo

    def proyectar_datos(self, n_samples=400):
        """Calcula el t-SNE basado en el género real de la DB."""
        docs = list(self.col.aggregate([
            {"$match": {"embeddings.beto_cls": {"$exists": True}}},
            {"$sample": {"size": n_samples}}
        ]))

        if not docs:
            return None

        vectores = np.array([d["embeddings"]["beto_cls"] for d in docs], dtype='float32')
        generos = [str(d.get("genero", "N/A")).title() for d in docs]
        titulos = [f"{d.get('titulo', '?')} - {d.get('artista', '?')}" for d in docs]

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        datos2d = tsne.fit_transform(vectores)

        return {
            'x': datos2d[:, 0].tolist(),
            'y': datos2d[:, 1].tolist(),
            'genero': generos,
            'info': titulos
        }

    def generar_grafico(self, data):
        # Crea el scatter plot coloreado por GÉNERO
        df = pd.DataFrame(data)
        fig = px.scatter(
            df, x='x', y='y', color='genero', hover_name='info',
            title="t-SNE BETO — Proyección de Géneros Reales",
            template="plotly_white" # Combinando con tu fondo blanco
        )
        return fig