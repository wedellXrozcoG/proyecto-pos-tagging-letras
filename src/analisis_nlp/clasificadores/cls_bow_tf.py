from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import plotly.express as px
import pandas as pd
from src.analisis_nlp.BoW.BagOfWords import BagOfWordsAnalyzer, TFIDFAnalyzer


class ClasificadorBowTfidf:
    """
    Clasifica géneros usando BoW y TF-IDF y genera
    métricas + gráfico comparativo para Dash.
    """

    def __init__(self, corpus: list[str], labels: list[str],
                 max_features: int = 50, test_size: float = 0.25):

        self.corpus = corpus
        self.labels = labels
        self.test_size = test_size

        self.bow = BagOfWordsAnalyzer(max_features=max_features)
        self.tfidf = TFIDFAnalyzer(max_features=max_features)

        self._cache = {}

    # =========================
    # ENTRENAMIENTO INTERNO
    # =========================
    def _entrenar_modelo(self, X, nombre_metodo: str):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            self.labels,
            test_size=self.test_size,
            random_state=42,
            stratify=self.labels
        )

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        reporte = classification_report(y_test, y_pred, output_dict=True)

        tabla = [
            {
                "Método": nombre_metodo,
                "Género": clase.capitalize(),
                "Precisión": round(metricas["precision"], 2),
                "Recall": round(metricas["recall"], 2),
                "F1-Score": round(metricas["f1-score"], 2),
                "Canciones": int(metricas["support"]),
            }
            for clase, metricas in reporte.items()
            if clase not in ["accuracy", "macro avg", "weighted avg"]
        ]

        return acc, tabla

    # =========================
    # MÉTOD PRINCIPAL DASH
    # =========================
    def obtener_reporte_para_dash(self):
        """
        Ejecuta entrenamiento y devuelve estructura serializable.
        """

        X_bow = self.bow.fit_transform(self.corpus).toarray()
        X_tfidf = self.tfidf.fit_transform(self.corpus).toarray()

        acc_bow, tabla_bow = self._entrenar_modelo(X_bow, "BoW")
        acc_tfidf, tabla_tfidf = self._entrenar_modelo(X_tfidf, "TF-IDF")

        self._cache = {
            "BoW": {"acc": acc_bow, "tabla": tabla_bow},
            "TF-IDF": {"acc": acc_tfidf, "tabla": tabla_tfidf}
        }

        return self._cache

    # =========================
    # VISUALIZACIÓN DASH
    # =========================
    def generar_componentes_visuales(self, cache):

        tabla_completa = []
        for metodo in cache.values():
            tabla_completa.extend(metodo["tabla"])

        df = pd.DataFrame(tabla_completa)

        fig = px.bar(
            df,
            x="Género",
            y="F1-Score",
            color="Método",
            barmode="group",
            text="F1-Score",
            title="F1-Score por Género — BoW vs TF-IDF"
        )

        fig.update_traces(textposition="outside")

        return df, fig