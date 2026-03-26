from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class CorpusLoader:
    """Carga el corpus desde MongoDB según géneros y tamaño de muestra."""

    def __init__(self, generos: list[str] = None, sample_size: int = 20):
        self.generos     = ["rock", "pop", "hip hop"]
        self.sample_size = sample_size
        self.corpus:  list[str] = []
        self.labels:  list[str] = []

    def load(self, col) -> tuple[list[str], list[str]]:
        self.corpus.clear()
        self.labels.clear()

        for genero in self.generos:
            docs = list(col.aggregate([
                {"$match": {"letra": {"$exists": True}, "genero": genero}},
                {"$sample": {"size": self.sample_size}}
            ]))
            for d in docs:
                self.corpus.append(d["letra"])
                self.labels.append(genero)

        print(f"Corpus: {len(self.corpus)} canciones — géneros: {set(self.labels)}\n")
        return self.corpus, self.labels


class BagOfWordsAnalyzer:
    """Representa y analiza el corpus con Bag of Words."""

    def __init__(self, max_features: int = 50):
        self.vectorizer = CountVectorizer(max_features=max_features)
        self.matrix     = None
        self.vocab      = None

    def fit_transform(self, corpus: list[str]):
        self.matrix = self.vectorizer.fit_transform(corpus)
        self.vocab  = self.vectorizer.get_feature_names_out()
        return self.matrix

    @property
    def sparsity(self) -> float:
        rows, cols = self.matrix.shape
        return (1 - self.matrix.nnz / (rows * cols)) * 100

    def summary(self, n_docs: int = 3, n_words: int = 15) -> pd.DataFrame:
        df = pd.DataFrame(
            self.matrix.toarray()[:n_docs, :n_words],
            columns=self.vocab[:n_words],
            index=[f"Doc {i+1}" for i in range(n_docs)]
        )
        print("=== BAG OF WORDS ===")
        print(f"Forma: {self.matrix.shape} | Dispersión: {self.sparsity:.1f}%")
        print(f"Vocabulario (primeras 15): {list(self.vocab[:15])}")
        print(f"\nPrimeros {n_docs} documentos × {n_words} palabras:")
        print(df.to_string())
        return df

    def demo_orthogonality(self, palabras: list[str] = None):
        palabras    = palabras or ["love", "fire", "money"]
        mini_bow    = CountVectorizer()
        mini_matrix = mini_bow.fit_transform(palabras)

        print("\nVectores one-hot de palabras individuales:")
        for i, palabra in enumerate(palabras):
            print(f"  '{palabra}': {mini_matrix[i].toarray().flatten()}")

        sim_01 = cosine_similarity(mini_matrix[0], mini_matrix[1])[0][0]
        sim_02 = cosine_similarity(mini_matrix[0], mini_matrix[2])[0][0]
        print(f"\nSim coseno '{palabras[0]}' vs '{palabras[1]}': {sim_01:.3f}")
        print(f"Sim coseno '{palabras[0]}' vs '{palabras[2]}': {sim_02:.3f}")
        print(f"\n⚠️  BoW trata '{palabras[1]}' y '{palabras[2]}' igual respecto a '{palabras[0]}'")
        print("   Esto es lo que los embeddings resuelven.")



from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class TFIDFAnalyzer:
    """Representa y analiza el corpus con TF-IDF."""

    def __init__(self, max_features: int = 50):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.matrix     = None
        self.vocab      = None

    def fit_transform(self, corpus: list[str]):
        self.matrix = self.vectorizer.fit_transform(corpus)
        self.vocab  = self.vectorizer.get_feature_names_out()
        return self.matrix

    @property
    def sparsity(self) -> float:
        rows, cols = self.matrix.shape
        return (1 - self.matrix.nnz / (rows * cols)) * 100

    def top_words_by_genre(self, labels: list[str], top_n: int = 5):
        """
        Calcula el promedio TF-IDF por género y muestra las top palabras reales.
        """

        if self.matrix is None:
            raise ValueError("Primero debes ejecutar fit_transform().")

        print("\n=== TF-IDF (Promedio por Género) ===")
        print(f"Forma: {self.matrix.shape} | Dispersión: {self.sparsity:.1f}%\n")

        labels_array = np.array(labels)
        generos_unicos = np.unique(labels_array)

        for genero in generos_unicos:
            indices = np.where(labels_array == genero)[0]

            # Promedio de vectores TF-IDF del género
            mean_vector = self.matrix[indices].mean(axis=0)
            mean_vector = np.asarray(mean_vector).flatten()

            # Top palabras
            top_indices = mean_vector.argsort()[-top_n:][::-1]
            top_words = [(self.vocab[i], f"{mean_vector[i]:.3f}") for i in top_indices]

            print(f"{genero.capitalize()}:")
            print(f"--> {top_words}\n")