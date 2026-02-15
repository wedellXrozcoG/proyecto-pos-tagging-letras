import matplotlib.pyplot as plt
import seaborn as sns

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

        # El resto del código se queda igual...
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points')

        plt.title(f'{titulo} {columna}', fontsize=15, fontweight='bold')
        plt.show()

    def grafico_pie(self, columna, titulo="Distribución"):
        """Genera un gráfico de pastel para una columna categórica."""
        plt.figure(figsize=(8, 8))
        counts = self.df[columna].value_counts()

        # Definir colores bonitos
        colors = sns.color_palette('pastel')[0:len(counts)]

        plt.pie(counts,
                labels=counts.index,
                autopct='%1.1f%%',
                startangle=140,
                colors=colors,
                wedgeprops={'edgecolor': 'white'})

        plt.title(f'{titulo}: {columna}', fontsize=15, fontweight='bold')
        plt.show()