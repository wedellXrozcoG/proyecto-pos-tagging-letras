import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import gc
from src.db_manager import get_collection
from src.analisis_nlp.clustering.clustering_bert import AnalizadorClustering
from src.analisis_nlp.clustering.clustering_word2vec import AnalizadorClusteringWord2Vec

layout = dbc.Container([
    dcc.Store(id='store-clustering-bert', storage_type='session'),
    dcc.Store(id='store-clustering-w2v', storage_type='session'),

    html.H2("Clustering K-Means + Silhouette Score", className="text-info mb-2"),
    dbc.Button("Ejecutar Bert", id="btn-clustering", color="danger", size="sm", className="mb-4"),
dbc.Button("Clustering Word2Vec", id="btn-clustering-w2v", color="primary", size="sm", className="mb-4"),
    dcc.Loading(type="circle", children=[
        html.Div(id="clustering-resultado-display"),
        html.Div(id="clustering-w2v-display")
    ])
], fluid=True)


# cálculo
@callback(
    Output("store-clustering-bert", "data"),
    Input("btn-clustering", "n_clicks"),
    prevent_initial_call=True
)
def calcular_clustering(n_clicks):
    col = get_collection()
    analizador = AnalizadorClustering(col)
    cache = analizador.ejecutar_analisis()

    del analizador
    gc.collect()
    return cache


@callback(
    Output("clustering-resultado-display", "children"),
    Input("store-clustering-bert", "data")
)
def mostrar_clustering(data): #clase del gráfico
    if not data:
        return html.Div("Haz clic en ejecutar para ver los clusters.", className="text-muted p-4")

    analizador_tool = AnalizadorClustering(None)
    fig = analizador_tool.generar_grafico(data)

    return [
        dbc.Alert(f"Silhouette Score BERT: {data['score']:.4f}", color="info", className="h4 text-center"),
        dcc.Graph(figure=fig)
    ]

#-----------------------------
#cálculo w2v
@callback(
    Output("store-clustering-w2v", "data"),
    Input("btn-clustering-w2v", "n_clicks"),
    prevent_initial_call=True
)
def calcular_clustering_w2v(n_clicks):
    col = get_collection()
    analizador = AnalizadorClusteringWord2Vec(col)
    cache = analizador.ejecutar_analisis()

    del analizador
    gc.collect()
    return cache

#gráfico w2v
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
