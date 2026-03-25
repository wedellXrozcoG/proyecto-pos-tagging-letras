import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px


class AnalizadorTSNEWord2Vec:
    """Genera proyecciones t-SNE para comparar la separación visual de géneros reales."""

    def __init__(self, col_mongo):
        self.col = col_mongo

    def ejecutar_proyeccion(self, n_samples=500):
        # 1. Extraemos una muestra que tenga el embedding y el género real
        docs = list(self.col.aggregate([
            {"$match": {
                "embeddings.word2vec_avg": {"$exists": True},
                "genero": {"$ne": None}
            }},
            {"$sample": {"size": n_samples}}
        ]))

        if not docs:
            return None

        # 2. Preparamos los datos
        # Extraemos los vectores de 100 dimensiones
        vectores = np.array([d["embeddings"]["word2vec_avg"] for d in docs], dtype='float32')

        # Guardamos metadatos para el hover del gráfico
        titulos = [d.get("titulo", "Sin título") for d in docs]
        artistas = [d.get("artista", "Desconocido") for d in docs]
        generos = [str(d.get("genero", "Otros")).capitalize() for d in docs]

        # 3. Reducción de Dimensionalidad con t-SNE
        # Perplexity define el balance entre la atención a datos locales y globales
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
        datos2d = tsne.fit_transform(vectores)

        # 4. Retornamos un DataFrame listo para Plotly
        return pd.DataFrame({
            'x': datos2d[:, 0],
            'y': datos2d[:, 1],
            'Genero Real': generos,
            'Cancion': titulos,
            'Artista': artistas
        })

    def generar_grafico(self, df_tsne):
        if df_tsne is None:
            return "No hay datos suficientes."

        # 5. Creación del gráfico interactivo (Requisito 5.3)
        fig = px.scatter(
            df_tsne,
            x='x', y='y',
            color='Genero Real',  # Aquí comparamos por el género de la base de datos
            hover_name='Cancion',
            hover_data=['Artista'],
            title="Proyección t-SNE: Separación Semántica de Géneros (Word2Vec)",
            template="plotly_dark",  # Tema oscuro para diseño profesional
            color_discrete_sequence=px.colors.qualitative.Vivid
        )

        # Mejoramos la estética del gráfico
        fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color='White')))
        fig.update_layout(
            legend_title_text='Géneros Musicales',
            xaxis_title="Dimensión t-SNE 1",
            yaxis_title="Dimensión t-SNE 2"
        )

        return fig