#aquí solo va el código clasificador bert
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import pandas as pd


class ClasificadorBert:
    """Clasifica géneros usando embeddings BERT [CLS] guardados en MongoDB."""

    GENEROS = ['hip hop', 'pop', 'rock']

    def __init__(self, col_mongo):
        self.col   = col_mongo
        self.model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        self.X     = None
        self.y     = None

    def preparar_datos(self):
        cursor = self.col.find(
            {"embeddings.beto_cls": {"$exists": True}},
            {"embeddings.beto_cls": 1, "genero": 1, "_id": 0}
        ).limit(6000)

        data_x, data_y = [], []
        for doc in cursor:
            gen = str(doc.get('genero', '')).lower().strip()
            if gen in self.GENEROS:
                data_x.append(doc['embeddings']['beto_cls'])
                data_y.append(gen)

        self.X = np.array(data_x, dtype='float32')
        self.y = np.array(data_y)

    def obtener_reporte_para_dash(self):
        """Entrena y devuelve accuracy + tabla para Dash."""
        if self.X is None:
            self.preparar_datos()

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        self.model.fit(X_train, y_train)
        y_pred       = self.model.predict(X_test)
        acc          = accuracy_score(y_test, y_pred)
        reporte_dict = classification_report(y_test, y_pred, output_dict=True)

        tabla = []
        for llave, metricas in reporte_dict.items():
            if llave not in ['accuracy', 'macro avg', 'weighted avg']:
                tabla.append({
                    "Género"   : llave.capitalize(),
                    "Precisión": round(metricas['precision'], 2),
                    "Recall"   : round(metricas['recall'], 2),
                    "F1-Score" : round(metricas['f1-score'], 2),
                    "Canciones": int(metricas['support'])
                })

        return acc, tabla

    def generar_componentes_visuales(self, cache):
        """Toma el cache y devuelve el gráfico y el DF ya listos."""
        df = pd.DataFrame(cache['tabla'])

        # Configuramos el gráfico aquí adentro
        fig = px.bar(
            df, x='Género', y='F1-Score', color='Género',
            title="F1-Score por Género — BERT"
        )

        return df, fig