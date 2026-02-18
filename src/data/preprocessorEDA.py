import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

class preprocesador:
    def __init__(self, df):
        self.df = df

    def mostrar_head(self):
        """Devuelve las primeras 5 filas del DataFrame"""
        return self.df.head()

    def mostrar_info(self):
        """Devuelve la información general del DataFrame"""
        return self.df.info()

    def mostrar_dimensiones(self):
        """Devuelve las dimensiones del DataFrame"""
        return self.df.shape

    def mostrar_valores_nulos(self):
        """Devuelve la cantidad de valores nulos por columna"""
        return self.df.isnull().sum()

    def resum_transpose(self):
        """Principales datos estadísticos del dataframe"""
        return self.df.describe().transpose()

    def ch_bool(self, columna_binaria):
        """Cambia datos binarios Yes/No a 1 y 0"""
        self.df[columna_binaria] = self.df[columna_binaria].replace({'Yes': 1, 'No': 0})
        return self.df

            #***************************Graficos*************************

    def grafico_barras(self, columna, titulo="Frecuencia por"):
        """Genera un gráfico de barras usando el dataframe de la clase."""
        plt.figure(figsize=(10, 6))

        ax = sns.countplot(
            data=self.df,
            x=columna,
            palette='viridis',
            hue=columna,
            legend=False
        )

        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points')

        plt.title(f'{titulo} {columna}', fontsize=15, fontweight='bold')
        plt.show()

    def grafico_pie(self, genero=None):
        """Genera un gráfico pastel con las emociones dependiendo del genero."""
        df = self.df.copy()
        if genero:
            df = df[df['Genre'].str.lower() == genero.lower()]

        # 2. Contamos las emociones
        conteo = df['emotion'].value_counts().reset_index()
        conteo.columns = ['emocion', 'total']

        # 3. Dibujar gráfico
        fig = px.pie(conteo,
                     values='total',
                     names='emocion',
                     title=f"Emociones en: {genero if genero else 'Todos los géneros'}",
                     hole=0.4)  # Estilo dona

        fig.show()

    def grafico_popularidad_genero(self):
        """Genera un boxplot segun el genero"""
        fig = px.box(self.df,
                     x="Genre",
                     y="Popularity",
                     color="Genre",
                     title="Popularidad según el Género Musical")
        fig.show()

    def grafico_popularidad_tiempo(self):
        """Muestra la distribucion segun la popularidad"""
        fig = px.scatter(self.df,
                         x="Release Date",
                         y="Popularity",
                         color="Genre",
                         hover_name="Explicit",  # Lo que se va a ver al pasar el cursor
                         title="Evolución de la Popularidad en el Tiempo")
        fig.show()