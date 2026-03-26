import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import gc
import pandas as pd
from src.db_manager import get_collection

# Importa tus clases existentes
from src.analisis_nlp.tsne.tsne_bert import VisualizadorTSNE
from src.analisis_nlp.tsne.tsne_word2vec import AnalizadorTSNEWord2Vec
from src.analisis_nlp.BoW.BagOfWords import CorpusLoader
from src.analisis_nlp.tsne.tsne_bow_tf import VisualizadorTSNEBowTfidf

layout = dbc.Container([
    # Stores
    dcc.Store(id='store-tsne-bert', storage_type='session'),
    dcc.Store(id='store-tsne-w2v', storage_type='session'),
    dcc.Store(id='store-tsne-bow', storage_type='session'),
    dcc.Store(id='store-tsne-tfidf', storage_type='session'),

    html.H2("Visualización t-SNE", className="text-info mb-2"),

    # Botones
    dbc.Button("Generar BERT", id="btn-tsne-bert", color="danger", className="mb-2"),
    dbc.Button("Generar Word2Vec", id="btn-tsne-w2v", color="primary", className="mb-2"),
    dbc.Button("Generar BoW", id="btn-tsne-bow", color="warning", className="mb-2"),
    dbc.Button("Generar TF-IDF", id="btn-tsne-tfidf", color="success", className="mb-2"),

    # Áreas de visualización
    dcc.Loading(type="graph", children=[
        html.Div(id="tsne-display-bert"),
        html.Div(id="tsne-display-w2v"),
        html.Div(id="tsne-display-bow"),
        html.Div(id="tsne-display-tfidf"),
    ])
], fluid=True)

# ------------------ CALLBACK BERT ------------------
@callback(
    Output("store-tsne-bert", "data"),
    Input("btn-tsne-bert", "n_clicks"),
    prevent_initial_call=True
)
def calcular_tsne_bert(n_clicks):
    col = get_collection()
    visualizador = VisualizadorTSNE(col)
    data = visualizador.proyectar_datos()
    del visualizador
    gc.collect()
    return data

@callback(
    Output("tsne-display-bert", "children"),
    Input("store-tsne-bert", "data")
)
def mostrar_tsne_bert(data):
    if not data:
        return html.Div("Haz clic en el botón para generar la proyección BERT.",
                        className="text-center p-5 text-muted")
    vis_tool = VisualizadorTSNE(None)
    fig = vis_tool.generar_grafico(data)
    return dcc.Graph(figure=fig)

# ------------------ CALLBACK Word2Vec ------------------
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

@callback(
    Output("tsne-display-w2v", "children"),
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

# ------------------ CALLBACK BoW ------------------
@callback(
    Output("store-tsne-bow", "data"),
    Input("btn-tsne-bow", "n_clicks"),
    prevent_initial_call=True
)
def calcular_tsne_bow(n_clicks):
    col = get_collection()
    loader = CorpusLoader()
    corpus, labels = loader.load(col)
    vis = VisualizadorTSNEBowTfidf(corpus=corpus, labels=labels, metodo='bow')
    data = vis.proyectar_datos()
    del vis, loader
    gc.collect()
    return data

@callback(
    Output("tsne-display-bow", "children"),
    Input("store-tsne-bow", "data")
)
def mostrar_tsne_bow(data):
    if not data:
        return html.Div("Genera BoW para comparar.",
                        className="text-center p-5 text-muted")
    vis_tool = VisualizadorTSNEBowTfidf(None, None, metodo='bow')
    fig = vis_tool.generar_grafico(data)
    return dcc.Graph(figure=fig)

# ------------------ CALLBACK TF-IDF ------------------
@callback(
    Output("store-tsne-tfidf", "data"),
    Input("btn-tsne-tfidf", "n_clicks"),
    prevent_initial_call=True
)
def calcular_tsne_tfidf(n_clicks):
    col = get_collection()
    loader = CorpusLoader()
    corpus, labels = loader.load(col)
    vis = VisualizadorTSNEBowTfidf(corpus=corpus, labels=labels, metodo='tfidf')
    data = vis.proyectar_datos()
    del vis, loader
    gc.collect()
    return data

@callback(
    Output("tsne-display-tfidf", "children"),
    Input("store-tsne-tfidf", "data")
)
def mostrar_tsne_tfidf(data):
    if not data:
        return html.Div("Genera TF-IDF para comparar.",
                        className="text-center p-5 text-muted")
    vis_tool = VisualizadorTSNEBowTfidf(None, None, metodo='tfidf')
    fig = vis_tool.generar_grafico(data)
    return dcc.Graph(figure=fig)