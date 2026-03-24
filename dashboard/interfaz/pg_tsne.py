import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import gc
from src.db_manager import get_collection
from src.analisis_nlp.tsne.tsne_bert import VisualizadorTSNE

layout = dbc.Container([
    dcc.Store(id='store-tsne-data', storage_type='session'),

    html.H2("Visualización t-SNE", className="text-info mb-2"),
    dbc.Button("Generar Visualización", id="btn-tsne", color="danger", className="mb-3"),

    dcc.Loading(type="graph", children=[
        html.Div(id="tsne-display-area")
    ])
], fluid=True)


@callback(
    Output("store-tsne-data", "data"),
    Input("btn-tsne", "n_clicks"),
    prevent_initial_call=True
)
def calcular_tsne(n_clicks):
    col = get_collection()
    visualizador = VisualizadorTSNE(col)
    cache = visualizador.proyectar_datos()

    del visualizador
    gc.collect()
    return cache


@callback(
    Output("tsne-display-area", "children"),
    Input("store-tsne-data", "data")
)
def mostrar_tsne(data):
    if not data:
        return html.Div("Haz clic en el botón para generar la proyección.",
                        className="text-center p-5 text-muted")

    vis_tool = VisualizadorTSNE(None)
    fig = vis_tool.generar_grafico(data)

    return dcc.Graph(figure=fig)