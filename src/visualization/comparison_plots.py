import pandas as pd
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class POSVisualizer:
    def __init__(self, df):
        self.df = df

    def comparar_distribucion(self):
        # Contar tags
        nltk_counts = Counter(self.df['nltk_tag'])
        spacy_counts = Counter(self.df['spacy_pos'])

        # Top 10
        nltk_top10 = nltk_counts.most_common(10)
        spacy_top10 = spacy_counts.most_common(10)

        nltk_names = [t[0] for t in nltk_top10]
        nltk_values = [t[1] for t in nltk_top10]

        spacy_names = [t[0] for t in spacy_top10]
        spacy_values = [t[1] for t in spacy_top10]

        # Subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Top 10 NLTK Tags", "Top 10 Spacy POS")
        )

        # NLTK
        fig.add_trace(
            go.Bar(
                x=nltk_names,
                y=nltk_values,
                name="NLTK"
            ),
            row=1,
            col=1
        )

        # Spacy
        fig.add_trace(
            go.Bar(
                x=spacy_names,
                y=spacy_values,
                name="Spacy"
            ),
            row=1,
            col=2
        )

        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            height=500,
            width=1000,
            title="Comparación de Distribución POS"
        )

        fig.update_xaxes(tickangle=45)

        return fig
