# 🎵 Análisis Morfosintáctico de Letras Musicales y Análisis Semántico de Letras Musicales

### Proyecto 1/2 — POS Tagging con NLTK / spaCy y Word2Vec / BETO (Spanish BERT)

**Curso:** Minería de Textos\
**Institución:** Colegio Universitario de Cartago (CUC)\
**Profesor:** Osvaldo Gonzalez Chaves

------------------------------------------------------------------------

## 📋 Descripción

*Proyecto 1:* Este proyecto aplica **POS Tagging (Part-of-Speech Tagging)** con NLTK y spaCy para analizar la estructura morfosintáctica de letras musicales. Comparamos patrones gramaticales entre géneros (Rock, Pop, Hip-Hop, Reggaetón) y exploramos cómo evoluciona el lenguaje musical a través del tiempo.

*Proyecto 2:* Este proyecto extiende el Proyecto 1 (Análisis Morfosintáctico con POS Tagging) incorporando técnicas avanzadas de representación semántica del lenguaje. Los estudiantes aplicarán Word2Vec y BETO (Spanish BERT) para descubrir relaciones semánticas profundas en letras musicales, comparando representaciones estáticas vs. contextuales. Además, integrarán MongoDB como sistema de almacenamiento NoSQL y enriquecerán su corpus mediante Web Scraping.

------------------------------------------------------------------------

## 🚀 Instalación

``` bash
# 1. Clonar el repositorio
git clone https://github.com/wedellXrozcoG/proyecto-pos-tagging-letras
cd proyecto-pos-tagging-letras

# 2. Descargar modelos de spaCy
python -m spacy download en_core_web_sm

# 3. (Opcional) Descargar modelo en español
python -m spacy download es_core_news_sm
```

------------------------------------------------------------------------

## 📁 Estructura del Proyecto basico

```         
proyecto-pos-tagging-letras/
├── data/               # Datos crudos, procesados y resultados
├── notebooks/          # Jupyter Notebooks del análisis paso a paso
├── src/                # Código fuente modular (paquetes Python)
├── scripts/            # Scripts ejecutables para correr el pipeline
├── dashboard/          # Aplicación Plotly Dash interactiva
├── tests/              # Pruebas unitarias
├── docs/               # Documentación del proyecto
└── outputs/            # Gráficos, tablas e informe final
```

------------------------------------------------------------------------

## ▶️ Cómo ejecutar

### Opción A: Notebooks paso a paso

``` bash
jupyter notebook notebooks/
```

Ejecuta los notebooks en orden (01 → 11).


------------------------------------------------------------------------

## 📊 Dataset

Usamos el dataset **"500K+ Spotify Songs with Lyrics,Emotions & More"** de Kaggle:\
🔗 <https://www.kaggle.com/datasets/devdope/900k-spotify?select=spotify_dataset.csv>

Para descargarlo automáticamente:

``` bash
python scripts/download_dataset.py
```
------------------------------------------------------------------------


### PROYECTO 2 — Análisis Semántico de Letras Musicales Aplicando Word2Vec, BETO y MongoDB
#### Estructura colección mongodb
```
{
   "_id": "ObjectId",
   "titulo": "Nombre de la canción",
   "artista": "Nombre del artista",
   "genero": "Rock" | "Pop" | "Hip-Hop" 
   "anio": 2015,
   "letra": "Texto completo de la letra...",
   "idioma": "es" | "en",
   "fuente": "kaggle" | "scraping",
   "url_fuente": "https://...",
   "fecha_recopilacion": "ISODate",
   "pos_tags": {
       "nltk": [...],
       "spacy": [...]
    },
    "embeddings": {
        "word2vec_avg": [...],
        "beto_cls": [...]
    },
   "metricas": {
       "num_palabras": 250,
       "densidad_lexica": 0.65,
       "ratio_sustantivos_verbos": 1.2
   }
}

```
------------------------------------------------------------------------

## 👥 Equipo

| Nombre          | GitHub         |
|-----------------|----------------|
| Gilary Granados | @Gilary001      |
| Wedell Orozco   | @wedellXrozcoG |
