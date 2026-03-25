import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import gc
import pandas as pd
from src.db_manager import get_collection
from src.analisis_nlp.tsne.tsne_bert import VisualizadorTSNE
from src.analisis_nlp.tsne.tsne_word2vec import AnalizadorTSNEWord2Vec

layout = dbc.Container([
    dcc.Store(id='store-tsne-data', storage_type='session'),
    dcc.Store(id='store-tsne-w2v', storage_type='session'),

    html.H2("Visualización t-SNE", className="text-info mb-2"),
    dbc.Button("Generar Bert", id="btn-tsne", color="danger", className="mb-3"),
    dbc.Button("Generar Word2Vec", id="btn-tsne-w2v", color="primary", className="mb-3"),

    dcc.Loading(type="graph", children=[
        html.Div(id="tsne-display-area"),
        html.Div(id="tsne-w2v-display-area"),
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

#--------------
#Calculo w2v
@callback(
    Output("store-tsne-w2v", "data"),
    Input("btn-tsne-w2v", "n_clicks"),
    prevent_initial_call=True
)
def calcular_tsne_w2v(n_clicks):
    col = get_collection()
    visualizador = AnalizadorTSNEWord2Vec(col)
    df = visualizador.ejecutar_proyeccion()

    del visualizador
    gc.collect()
    return df.to_dict("records")


#grafico w2v
@callback(
    Output("tsne-w2v-display-area", "children"),
    Input("store-tsne-w2v", "data")
)
def mostrar_tsne_w2v(data):
    if not data:
        return html.Div("Genera Word2Vec para comparar.",
                        className="text-center p-5 text-muted")

    df = pd.DataFrame(data)
    vis_tool = AnalizadorTSNEWord2Vec(None)
    fig = vis_tool.generar_grafico(df)

    return dcc.Graph(figure=fig)