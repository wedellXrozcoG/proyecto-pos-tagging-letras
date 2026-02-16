# 🎵 Análisis Morfosintáctico de Letras Musicales

### Proyecto 1 — POS Tagging con NLTK y spaCy

**Curso:** Minería de Textos\
**Institución:** Colegio Universitario de Cartago (CUC)\
**Profesor:** Osvaldo Gonzalez Chaves

------------------------------------------------------------------------

## 📋 Descripción

Este proyecto aplica **POS Tagging (Part-of-Speech Tagging)** con NLTK y spaCy para analizar la estructura morfosintáctica de letras musicales. Comparamos patrones gramaticales entre géneros (Rock, Pop, Hip-Hop, Reggaetón) y exploramos cómo evoluciona el lenguaje musical a través del tiempo.

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

## 📁 Estructura del Proyecto

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

Ejecuta los notebooks en orden (01 → 07).

### Opción B: Pipeline completo automático

``` bash
python scripts/preprocess_all.py
python scripts/run_pos_tagging.py
python scripts/generate_metrics.py
```

------------------------------------------------------------------------

## 📊 Dataset

Usamos el dataset **"500K+ Spotify Songs with Lyrics,Emotions & More"** de Kaggle:\
🔗 <https://www.kaggle.com/datasets/devdope/900k-spotify?select=spotify_dataset.csv>

Para descargarlo automáticamente:

``` bash
python scripts/download_dataset.py
```

------------------------------------------------------------------------

## 🔬 Hallazgos Principales

------------------------------------------------------------------------

## 👥 Equipo

| Nombre          | GitHub         |
|-----------------|----------------|
| Gilary Granados | @Gilary001      |
| Wedell Orozco   | @wedellXrozcoG |
