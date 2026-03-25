from gensim.models import Word2Vec
import spacy
nlp = spacy.load("en_core_web_sm")
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Funcion que hace limpieza en caso de ser necesario
def limpieza_para_word2vec(texto):
    if not texto: return []
    doc = nlp(texto.lower())

    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop  # Stopwords
           and not token.is_punct  # Quita puntos y comas
           and token.is_alpha  # Quita números/fechas
           and len(token.text) > 2  # <--- NUEVO: Quita letras sueltas (deletreo)
    ]
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

# -- Funcion que muestra que tanta similitud existe en distintos generos --
def similitud_generos(generos_data):
    nombres  = list(generos_data.keys())
    vectores = [
        np.mean([modelo.wv[p] for doc in corpus for p in doc if p in modelo.wv], axis=0)
        for _, (modelo, corpus) in generos_data.items()
    ]

    # Tabla de similitudes
    print(f"  {'Género 1':<15} {'Género 2':<15} {'Similitud':<12} {'Interpretación'}")
    print("  " + "-" * 65)
    for i in range(len(nombres)):
        for j in range(i+1, len(nombres)):
            sim = cosine_similarity([vectores[i]], [vectores[j]])[0][0]
            if sim > 0.8:
                interp = "Muy alta"
            elif sim > 0.6:
                interp = "Alta"
            elif sim > 0.4:
                interp = "Media"
            else:
                interp = "Baja"
            print(f"  {nombres[i]:<15} {nombres[j]:<15} {float(sim):<12.4f} {interp}")


    sns.heatmap(cosine_similarity(vectores), annot=True, fmt=".3f",
                xticklabels=nombres, yticklabels=nombres,
                cmap="YlOrRd", vmin=0, vmax=1)
    plt.title("Similitud entre géneros (Word2Vec)")
    plt.tight_layout()
    plt.show()

# -- Función para encontrar cuales son las palabras mas frecuentes en cada genero --
def vocabulario_genero(modelos_por_genero, top_n=10):

    for genero, modelo in modelos_por_genero.items():
        # Toma las 100 palabras más frecuentes del género actual
        top_actual = modelo.wv.index_to_key[:100]

        # Crea una lista de las palabras frecuentes de los OTROS géneros
        otros_tops = []
        for g_otro, m_otro in modelos_por_genero.items():
            if g_otro != genero:
                otros_tops.extend(m_otro.wv.index_to_key[:100])

        # Filtra: Palabras que están en el Top 100 pero NO en el de los demás
        palabras_relevantes = [p for p in top_actual if p not in otros_tops]

        print(f"\n=== Palabras clave que definen al {genero.upper()} ===")
        if palabras_relevantes:
            # Mostramos las top_n encontradas
            for i, palabra in enumerate(palabras_relevantes[:top_n], 1):
                # Extraemos la frecuencia real para darle más valor al print
                frecuencia = modelo.wv.get_vecattr(palabra, "count")
                print(f"{i}. {palabra} (Aparece {frecuencia} veces)")
        else:
            print("No se encontraron palabras suficientemente distintivas en el Top 100.")

# --Funcion para optener el promedio vectores--
def obtener_vector_promedio(letra, modelo):
    # Limpiamos la letra
    tokens = limpieza_para_word2vec(letra)

    # Extraemos los vectores de las palabras que el modelo conoce
    vectores = [modelo.wv[p] for p in tokens if p in modelo.wv]

    # Si hay vectores, promediamos si no, devolvemos lista vacía
    if vectores:
        return np.mean(vectores, axis=0).tolist()
    return []


# --Funcion para actualizar el mongodb (word2vec_avg)--
def actualizar_embeddings_mongo(coleccion, modelo):
    # Traemos las canciones (solo ID y letra)
    canciones = list(coleccion.find({}, {"_id": 1, "letra": 1}))
    print(f"Procesando {len(canciones)} canciones...")

    contador = 0
    for cancion in canciones:
        # Llamamos a la función de cálculo
        vector = obtener_vector_promedio(cancion.get("letra", ""), modelo)

        # Guardamos en la base de datos
        coleccion.update_one(
            {"_id": cancion["_id"]},
            {"$set": {"embeddings.word2vec_avg": vector}}
        )
        contador += 1

    print(f"¡Éxito! Se actualizaron {contador} documentos.")



# --Funcion para evaluar entre distintas palabras cuales tiene relacion/aproximidad--
def graficar_semantica_word2vec(modelo, grupos_semanticos, titulo="Relaciones Semánticas"):

    # 1. Configuración de colores (puedes añadir más si tienes más grupos)
    colores_lista = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71", "#9b59b6", "#1abc9c", "#34495e"]
    colores = {cat: colores_lista[i % len(colores_lista)] for i, cat in enumerate(grupos_semanticos.keys())}

    palabras_finales = []
    vectores_finales = []
    categorias_finales = []

    # 2. Extracción de vectores
    for categoria, palabras in grupos_semanticos.items():
        for p in palabras:
            if p in modelo.wv:
                palabras_finales.append(p)
                vectores_finales.append(modelo.wv[p])
                categorias_finales.append(categoria)
            else:
                print(f"Advertencia: '{p}' no encontrada en el vocabulario.")

    if not vectores_finales:
        print("Error: No se encontraron palabras del diccionario en el modelo.")
        return

    # 3. Reducción de dimensiones
    X = np.array(vectores_finales)
    # Perplexity ajustada al tamaño de los datos
    perp = min(5, len(X) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(X)

    # 4. Construcción del gráfico
    plt.figure(figsize=(12, 8))

    for cat in grupos_semanticos.keys():
        indices = [i for i, c in enumerate(categorias_finales) if c == cat]
        if not indices: continue

        plt.scatter(X_2d[indices, 0], X_2d[indices, 1],
                    c=colores[cat], label=cat, s=180, edgecolors='black', alpha=0.8)

        for i in indices:
            plt.annotate(palabras_finales[i], (X_2d[i, 0], X_2d[i, 1]),
                         fontsize=10, fontweight='bold', xytext=(7, 3),
                         textcoords='offset points')

    plt.title(titulo, fontsize=16, fontweight='bold', pad=20)
    plt.legend(title="Categorías", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.15, linestyle='--')
    plt.tight_layout()
    plt.show()