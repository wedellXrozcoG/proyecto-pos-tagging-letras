import torch
import numpy as np
import gc
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class BuscadorSemantico:

    def __init__(self, col_mongo):
        self.col           = col_mongo
        self.nombre_modelo = "bert-base-uncased"
        self.tokenizer     = AutoTokenizer.from_pretrained(self.nombre_modelo)
        self.model         = AutoModel.from_pretrained(self.nombre_modelo)
        self.model.eval()

    def _vectorizar_consulta(self, texto):
        """Convierte la consulta en un vector de 768D"""
        inputs = self.tokenizer(
            texto, return_tensors="pt",
            padding=True, truncation=True, max_length=64
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        vector = outputs.last_hidden_state[:, 0, :].numpy()
        del outputs, inputs
        return vector

    def buscar_similares(self, query_texto, n_resultados=10):
        """Busca las canciones más similares semánticamente a la consulta."""

        vector_query = self._vectorizar_consulta(query_texto)

        # $sample aleatoriza los documentos — evita traer siempre los mismos
        docs = list(self.col.aggregate([
            {"$match": {"embeddings.beto_cls": {"$exists": True}}},
            {"$sample": {"size": 1500}},  # pool grande y aleatorio
            {"$project": {"titulo": 1, "artista": 1, "genero": 1,
                          "embeddings.beto_cls": 1, "_id": 0}}
        ]))

        if not docs:
            return []

        vectores_db = np.array([d["embeddings"]["beto_cls"] for d in docs], dtype="float32")
        similitudes = cosine_similarity(vector_query, vectores_db)[0]

        resultados = sorted([
            {
                "Titulo"   : docs[i].get("titulo", "Unknown"),
                "Artista"  : docs[i].get("artista", "Unknown"),
                "Género"   : docs[i].get("genero", "N/A"),
                "Similitud": round(float(similitudes[i]) * 100, 2)
            }
            for i in range(len(docs))
        ], key=lambda x: x["Similitud"], reverse=True)

        del vectores_db, docs
        gc.collect()

        # Formatear el % solo para mostrar, no para ordenar
        for r in resultados[:n_resultados]:
            r["Similitud"] = f"{r['Similitud']}%"

        return resultados[:n_resultados]