import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import gc
from src.db_manager import get_collection
from src.analisis_nlp.clustering.clustering_bert import AnalizadorClustering
from src.analisis_nlp.clustering.clustering_word2vec import AnalizadorClusteringWord2Vec
from src.analisis_nlp.clustering.clustering_bow_tf import AnalizadorClusteringBowTfidf

layout = dbc.Container([
    dcc.Store(id='store-clustering-bert', storage_type='session'),
    dcc.Store(id='store-clustering-w2v', storage_type='session'),
    dcc.Store(id='store-clustering-bow', storage_type='session'),
    dcc.Store(id='store-clustering-tfidf', storage_type='session'),

    html.H2("Clustering K-Means + Silhouette Score", className="text-info mb-2"),
    dbc.Button("Ejecutar Bert", id="btn-clustering-bert", color="danger", size="sm", className="mb-2"),
    dbc.Button("Clustering Word2Vec", id="btn-clustering-w2v", color="primary", size="sm", className="mb-2"),
    dbc.Button("Clustering BoW", id="btn-clustering-bow", color="secondary", size="sm", className="mb-2"),
    dbc.Button("Clustering TF-IDF", id="btn-clustering-tfidf", color="success", size="sm", className="mb-2"),

    dcc.Loading(type="circle", children=[
        html.Div(id="clustering-resultado-display"),
        html.Div(id="clustering-w2v-display"),
        html.Div(id="clustering-bow-display"),
        html.Div(id="clustering-tfidf-display")
    ])
], fluid=True)

# -------------------------
# BERT
@callback(
    Output("store-clustering-bert", "data"),
    Input("btn-clustering-bert", "n_clicks"),
    prevent_initial_call=True
)
def calcular_clustering_bert(n_clicks):
    col = get_collection()
    analizador = AnalizadorClustering(col)
    cache = analizador.ejecutar_analisis()
    del analizador; gc.collect()
    return cache

@callback(
    Output("clustering-resultado-display", "children"),
    Input("store-clustering-bert", "data")
)
def mostrar_clustering_bert(data):
    if not data:
        return html.Div("Haz clic en ejecutar para ver los clusters.", className="text-muted p-4")
    analizador_tool = AnalizadorClustering(None)
    fig = analizador_tool.generar_grafico(data)
    return [
        dbc.Alert(f"Silhouette Score BERT: {data['score']:.4f}", color="info", className="h4 text-center"),
        dcc.Graph(figure=fig)
    ]

# -------------------------
# Word2Vec
@callback(
    Output("store-clustering-w2v", "data"),
    Input("btn-clustering-w2v", "n_clicks"),
    prevent_initial_call=True
)
def calcular_clustering_w2v(n_clicks):
    col = get_collection()
    analizador = AnalizadorClusteringWord2Vec(col)
    cache = analizador.ejecutar_analisis()
    del analizador; gc.collect()
    return cache

@callback(
    Output("clustering-w2v-display", "children"),
    Input("store-clustering-w2v", "data")
)
def mostrar_clustering_w2v(data):
    if not data:
        return html.Div("Ejecuta Word2Vec para ver los clusters.", className="text-muted p-4")
    analizador_tool = AnalizadorClusteringWord2Vec(None)
    fig = analizador_tool.generar_grafico(data)
    return [
        dbc.Alert(f"Silhouette Score Word2Vec: {data['score']:.4f}", color="primary", className="h4 text-center"),
        dcc.Graph(figure=fig)
    ]

# -------------------------
# BoW
@callback(
    Output("store-clustering-bow", "data"),
    Input("btn-clustering-bow", "n_clicks"),
    prevent_initial_call=True
)
def calcular_clustering_bow(n_clicks):
    col = get_collection()
    analizador = AnalizadorClusteringBowTfidf(col, max_features=50)
    cache = analizador.ejecutar_analisis(metodo='bow')
    del analizador; gc.collect()
    return cache

@callback(
    Output("clustering-bow-display", "children"),
    Input("store-clustering-bow", "data")
)
def mostrar_clustering_bow(data):
    if not data:
        return html.Div("Ejecuta BoW para ver los clusters.", className="text-muted p-4")
    analizador_tool = AnalizadorClusteringBowTfidf(None)
    fig = analizador_tool.generar_grafico(data, metodo='BoW')
    return [
        dbc.Alert(f"Silhouette Score BoW: {data['score']:.4f}", color="secondary", className="h4 text-center"),
        dcc.Graph(figure=fig)
    ]

# -------------------------
# TF-IDF
@callback(
    Output("store-clustering-tfidf", "data"),
    Input("btn-clustering-tfidf", "n_clicks"),
    prevent_initial_call=True
)
def calcular_clustering_tfidf(n_clicks):
    col = get_collection()
    analizador = AnalizadorClusteringBowTfidf(col, max_features=50)
    cache = analizador.ejecutar_analisis(metodo='tfidf')
    del analizador; gc.collect()
    return cache

@callback(
    Output("clustering-tfidf-display", "children"),
    Input("store-clustering-tfidf", "data")
)
def mostrar_clustering_tfidf(data):
    if not data:
        return html.Div("Ejecuta TF-IDF para ver los clusters.", className="text-muted p-4")
    analizador_tool = AnalizadorClusteringBowTfidf(None)
    fig = analizador_tool.generar_grafico(data, metodo='TF-IDF')
    return [
        dbc.Alert(f"Silhouette Score TF-IDF: {data['score']:.4f}", color="success", className="h4 text-center"),
        dcc.Graph(figure=fig)
    ]