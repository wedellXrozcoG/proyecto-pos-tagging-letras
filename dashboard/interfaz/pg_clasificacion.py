# pg_clasificacion.py

import dash
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
import gc
from src.db_manager import get_collection
from src.analisis_nlp.clasificadores.cls_bert import ClasificadorBert
from src.analisis_nlp.clasificadores.cls_word2vec import ClasificadorWord2Vec
from src.analisis_nlp.BoW.BagOfWords import CorpusLoader
from src.analisis_nlp.clasificadores.cls_bow_tf import ClasificadorBowTfidf

# ==============================
# LAYOUT PRINCIPAL
# ==============================
layout = dbc.Container([
    # Stores
    dcc.Store(id='store-bert', storage_type='session'),
    dcc.Store(id='store-w2v', storage_type='session'),
    dcc.Store(id='store-bowtfidf', storage_type='session'),

    html.H2("Clasificación de Género", className="text-info mb-2"),

    # Botones
    dbc.Button("Ejecutar Bert", id="btn-bert", color="danger", size="sm", className="mb-4"),
    dbc.Button("Ejecutar Word2Vec", id="btn-w2v", color="primary", size="sm", className="mb-4"),
    dbc.Button("Ejecutar BoW/TF-IDF", id="btn-bowtfidf", color="warning", size="sm", className="mb-4"),

    # Áreas de visualización
    dcc.Loading(type="circle", children=[
        html.Div(id="bert-display-area"),
        html.Div(id="w2v-display-area"),
        html.Div(id="bowtfidf-display-area")
    ])
], fluid=True)

# ==============================
# CALLBACKS BERT
# ==============================
@callback(
    Output("store-bert", "data"),
    Input("btn-bert", "n_clicks"),
    prevent_initial_call=True
)
def entrenar_bert(n_clicks):
    col = get_collection()
    cls_obj = ClasificadorBert(col)
    acc, tabla = cls_obj.obtener_reporte_para_dash()
    del cls_obj
    gc.collect()
    return {'acc': acc, 'tabla': tabla}

@callback(
    Output("bert-display-area", "children"),
    Input("store-bert", "data")
)
def renderizar_bert(cache):
    if cache is None:
        return html.Div("Presione el botón para iniciar el entrenamiento.", className="text-muted text-center p-5")
    cls_tool = ClasificadorBert(None)
    df, fig = cls_tool.generar_componentes_visuales(cache)
    return [
        dbc.Alert(f"Accuracy BERT: {cache['acc']:.2%}", color="success", className="h4 text-center"),
        dbc.Row([
            dbc.Col(dash_table.DataTable(
                data=cache['tabla'],
                columns=[{"name": i, "id": i} for i in df.columns],
                style_header={'backgroundColor': 'black', 'color': 'white'},
                style_cell={'textAlign': 'center'}
            ), width=5),
            dbc.Col(dcc.Graph(figure=fig), width=7),
        ])
    ]

# ==============================
# CALLBACKS WORD2VEC
# ==============================
@callback(
    Output("store-w2v", "data"),
    Input("btn-w2v", "n_clicks"),
    prevent_initial_call=True
)
def entrenar_w2v(n_clicks):
    col = get_collection()
    cls_obj = ClasificadorWord2Vec(col)
    acc, tabla = cls_obj.obtener_reporte_para_dash()
    del cls_obj
    gc.collect()
    return {'acc': acc, 'tabla': tabla}

@callback(
    Output("w2v-display-area", "children"),
    Input("store-w2v", "data")
)
def renderizar_w2v(cache):
    if cache is None:
        return html.Div("Ejecute Word2Vec para ver resultados.", className="text-muted text-center p-5")
    cls_tool = ClasificadorWord2Vec(None)
    df, fig = cls_tool.generar_componentes_visuales(cache)
    return [
        dbc.Alert(f"Accuracy Word2Vec: {cache['acc']:.2%}", color="primary", className="h4 text-center"),
        dbc.Row([
            dbc.Col(dash_table.DataTable(
                data=cache['tabla'],
                columns=[{"name": i, "id": i} for i in df.columns],
                style_header={'backgroundColor': 'black', 'color': 'white'},
                style_cell={'textAlign': 'center'}
            ), width=5),
            dbc.Col(dcc.Graph(figure=fig), width=7),
        ])
    ]

# ==============================
# CALLBACKS BoW/TF-IDF
# ==============================
@callback(
    Output("store-bowtfidf", "data"),
    Input("btn-bowtfidf", "n_clicks"),
    prevent_initial_call=True
)
def entrenar_bowtfidf(n_clicks):
    col = get_collection()
    loader = CorpusLoader(sample_size=20)
    corpus, labels = loader.load(col)
    cls_obj = ClasificadorBowTfidf(corpus, labels)
    cache = cls_obj.obtener_reporte_para_dash()
    del cls_obj
    gc.collect()
    return cache

@callback(
    Output("bowtfidf-display-area", "children"),
    Input("store-bowtfidf", "data")
)
def renderizar_bowtfidf(cache):
    if cache is None:
        return html.Div("Ejecute BoW/TF-IDF para ver resultados.", className="text-muted text-center p-5")
    cls_tool = ClasificadorBowTfidf([], [])
    df, fig = cls_tool.generar_componentes_visuales(cache)
    return [
        dbc.Alert(
            f"Accuracy BoW: {cache['BoW']['acc']:.2%} | Accuracy TF-IDF: {cache['TF-IDF']['acc']:.2%}",
            color="warning",
            className="h4 text-center"
        ),
        dbc.Row([
            dbc.Col(dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_header={'backgroundColor': 'black', 'color': 'white'},
                style_cell={'textAlign': 'center'}
            ), width=5),
            dbc.Col(dcc.Graph(figure=fig), width=7),
        ])
    ]