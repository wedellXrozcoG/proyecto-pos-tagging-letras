import torch
import random
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class Bert:
    def __init__(self, col_mongo):

        print("Cargando BERT")
        self.col = col_mongo
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.fill_mask = pipeline("fill-mask", model="bert-base-uncased", device=-1) #device -1 para evitar problemas con la compu

    def obtener_embedding_token(self, frase, palabra, tokenizer, model):
        inputs = tokenizer(frase, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)

        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        idx = -1
        for i, t in enumerate(tokens):
            if t.lower() == palabra.lower() or t.lower() == f"##{palabra.lower()}":
                idx = i
                break

        if idx != -1:
            return outputs.last_hidden_state[0][idx].numpy(), tokens
        return None, None

    def ejecutar_polisemia(self):
        lista_palabras = ["die", "beat", "burn", "baby", "love"] #palabras a analizar
        muestra = 8
        col = self.col
        tokenizer = self.tokenizer
        model = self.model

        for palabra in lista_palabras:
            print(f"\n{'-' * 50}")
            print(f"ANALIZANDO: {palabra.upper()}")

            pipeline_mongo = [
                {"$match": {"letra": {"$regex": rf"\b{palabra}\b", "$options": "i"}}},
                {"$sample": {"size": muestra}}
            ]

            frases, embeddings, tokens_list, info = [], [], [], []

            for doc in col.aggregate(pipeline_mongo):
                if len(frases) == 2: break

                texto = doc['letra'].replace('\n', ' ')
                palabras = texto.split()
                idx = next((i for i, w in enumerate(palabras) if palabra.lower() in w.lower().strip(",.!?\"")), -1)

                if idx == -1: continue

                frase = " ".join(palabras[max(0, idx - 12):min(len(palabras), idx + 13)])

                try:
                    # Aquí llama a la función de arriba
                    emb, tokens = self.obtener_embedding_token(frase, palabra, tokenizer, model)
                except Exception:
                    continue

                if emb is None: continue

                frases.append(frase)
                embeddings.append(emb)
                tokens_list.append(tokens)
                info.append(
                    f"[{doc.get('genero', 'N/A').upper()}] {doc.get('titulo', '?')} - {doc.get('artista', '?')}")

            if len(embeddings) == 2:
                for i in range(2):
                    print(f"\n  Canción {i + 1}: {info[i]}")
                    print(f"  Contexto : \"...{frases[i]}...\"")
                    print(f"  Tokens   : {tokens_list[i][:15]}...")

                sim = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
                print(f"\n  Similitud coseno (BERT): {sim:.4f}")
                print("💚 Resultado: Uso Semejante" if sim > 0.65 else "❤️ Resultado: Uso Polisémico")
            else:
                print(f"No se encontraron suficientes ejemplos para '{palabra}'")

    def ejecutar_mlm(self):
        col = self.col
        fill_mask = self.fill_mask

        def mlm_canciones(genero_objetivo):
            pipeline_mongo = [
                {"$match": {"genero": {"$regex": f"^{genero_objetivo}$", "$options": "i"}}},
                {"$sample": {"size": 1}}
            ]
            resultados = list(col.aggregate(pipeline_mongo))
            if not resultados: return

            doc = resultados[0]
            texto = doc['letra'].replace('\r', '')
            lineas = [l.strip() for l in texto.split('\n') if len(l.strip().split()) > 5]

            if not lineas:
                palabras_totales = texto.split()
                lineas = [" ".join(palabras_totales[i:i + 10]) for i in range(0, len(palabras_totales), 10)]

            frase_seleccionada = random.choice(lineas)
            palabras = frase_seleccionada.split()[:30]
            idx_to_mask = random.randint(0, len(palabras) - 1)
            palabra_oculta = palabras[idx_to_mask].strip(",.!?\"") #que no elija ninguna de esos signos como top 3 predicciones
            palabras[idx_to_mask] = "[MASK]"
            frase_enmascarada = " ".join(palabras)

            print(f"\nGÉNERO: {genero_objetivo.upper()}")
            print(f"{doc.get('titulo')} - {doc.get('artista')}")
            print(f"Frase corta: \"{frase_enmascarada}\"")

            predicciones = fill_mask(frase_enmascarada)
            print(f"¿Qué palabra falta? (Original: '{palabra_oculta}')")

            mostrados = 0 #contador top 3
            for p in predicciones:
                token = p['token_str'].strip()
                if token in [".", ",", "!", "?"]: continue
                match = "💚" if token.lower() == palabra_oculta.lower() else ""
                print(f"   {mostrados + 1}. {token} ({p['score']:.3f}) {match}")
                mostrados += 1
                if mostrados >= 3: break #top 3
            print("-" * 40)

        for g in ["rock", "pop", "hip hop"]:
            mlm_canciones(g)