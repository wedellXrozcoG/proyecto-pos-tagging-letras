from gensim.models import Word2Vec
import spacy
nlp = spacy.load("en_core_web_sm")

def limpieza_para_word2vec(texto):
    doc = nlp(texto.lower())
    tokens = [token.lemma_ for token in doc
              if not token.is_stop and not token.is_punct and token.is_alpha]
    return tokens



# --Funcion para el Entrenamiento de modelos CBOW y Skip-Gram--
def modelos_CBOW_SKIPGRAM(corpus, size, window, sg, min_count, workers):
    modelo = Word2Vec(sentences=corpus, #Corpus tokenizado
                      vector_size=size, #Dimenciones de las palabras
                      window=window, #Distancia de palabras n palabras hacia atrás y n hacia adelante.
                      sg=sg, #Algoritmo de aprendizaje 0=CBOW y 1=Skip-Gram
                      min_count=min_count, #Si una palabra aparece n veces la ignora
                      workers=workers) #nucleos de procesador
    return modelo


# --Funcion para traer solo las letras de un genero o todo completo--
def obtener_corpus(data, genero_corpus=None):
    corpus = []

    if genero_corpus is None:
        datos_a_procesar = data['letra']
        print("Procesando el corpus completo (todos los géneros)...")
    else:
        datos_a_procesar = data[data["genero"] == genero_corpus]['letra']
        print(f"Procesando corpus: {genero_corpus}...")

    for letra in datos_a_procesar:
        palabras_limpias = limpieza_para_word2vec(letra)
        corpus.append(palabras_limpias)

    return corpus


# --Funcion para realizar exploración de campos semánticos por género--
def exploracion_semantica(modelo_cbow, modelo_sg, genero, palabra, n):
    print(f"=== Campo semántico para: {genero.upper()} ===")
    print(f"Palabra clave: '{palabra}'\n")

    # Ejecución CBOW
    print(f"--- Resultados con CBOW ---")
    try:
        vecinos_c = modelo_cbow.wv.most_similar(palabra, topn=n)
        for p, sim in vecinos_c:
            print(f"{p}: {sim:.4f}")
    except:
        print(f"La palabra '{palabra}' no tiene coincidencias.")

    print("=" * 50)

    # Ejecución SKIP-GRAM
    print(f"--- Resultados con SKIP-GRAM ---")
    try:
        vecinos_s = modelo_sg.wv.most_similar(palabra, topn=n)
        for p, sim in vecinos_s:
            print(f"{p}: {sim:.4f}")
    except:
        print(f"La palabra '{palabra}' no tiene coincidencias.")

# --Funcion para realizar exploración de campos semánticos por género--
def analogia_vectoriales(modelo_cbow, modelo_sg, a, b, c, top_n=3):
    print(f"Analogía(CBOW): {a} es a {b} como {c} es a...")
    try:
        # Positive son los que suman (b y c), negative el que resta (a)
        resultados = modelo_cbow.wv.most_similar(positive=[b, c], negative=[a], topn=top_n)

        for i, (palabra, sim) in enumerate(resultados):
            marcador = "  *" if i == 0 else "   "
            print(f"{marcador} {palabra} (similitud: {sim:.4f})")
    except KeyError as e:
        print(f"Error: La palabra {e} no está en este modelo.")
    print("=" * 50)


    print(f"Analogía(SKIP-GRAM): {a} es a {b} como {c} es a...")
    try:
        resultados = modelo_sg.wv.most_similar(positive=[b, c], negative=[a], topn=top_n)

        for i, (palabra, sim) in enumerate(resultados):
            marcador = "  *" if i == 0 else "   "
            print(f"{marcador} {palabra} (similitud: {sim:.4f})")
    except KeyError as e:
        print(f"Error: La palabra {e} no está en este modelo.")
    print("=" * 50)