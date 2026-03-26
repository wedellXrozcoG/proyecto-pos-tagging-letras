Estructura_Documento_Cancion/
│
├── _id                     # Identificador único de MongoDB (ObjectID)
├── titulo                  # Nombre de la canción (String)
├── artista                 # Nombre del intérprete o banda (String)
├── genero                  # Estilo musical (Rock, Pop, etc.)
├── anio                    # Año de lanzamiento original (Int)
├── letra                   # Texto completo de la lírica (String)
├── idioma                  # Código de idioma (ej: "en", "es")
├── fuente                  # Método de obtención (ej: "scraping", "kaggle")
├── url_fuente              # Enlace directo al origen de la letra (String)
├── fecha_recopilacion      # Timestamp de captura (ISODate)
│
├── pos_tags/               # Objeto con etiquetas de Partes del Discurso
│   ├── nltk                # Lista de tokens etiquetados por NLTK (Array)
│   └── spacy               # Lista de tokens etiquetados por spaCy (Array)
│
├── embeddings/             # Representaciones vectoriales para ML
│   ├── beto_cls            # Vector de 768 dimensiones (BETO)
│   └── word2vec_avg        # Vector de 100 dimensiones (Word2Vec)
│
└── metricas/               # Análisis estadístico del texto
    ├── num_palabras        # Conteo total de palabras (Int)
    ├── densidad_lexica     # Relación palabras únicas / total (Float)
    └── ratio_sustantivos_verbos # Proporción entre nombres y acciones (Float)