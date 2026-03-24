import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import gc
from src.db_manager import get_collection
from src.analisis_nlp.clustering.clustering_bert import AnalizadorClustering

layout = dbc.Container([
    dcc.Store(id='store-clustering-bert', storage_type='session'),

    html.H2("Clustering K-Means + Silhouette Score", className="text-info mb-2"),
    dbc.Button("Ejecutar Análisis", id="btn-clustering", color="danger", size="sm", className="mb-4"),

    dcc.Loading(type="circle", children=[
        html.Div(id="clustering-resultado-display")
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