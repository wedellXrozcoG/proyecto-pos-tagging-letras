# aquí se llama también el w2v (solo ejecución) Aquí va tanto bert como w2v

import dash
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
import gc
from src.db_manager import get_collection
from src.analisis_nlp.clasificadores.cls_bert import ClasificadorBert
from src.analisis_nlp.clasificadores.cls_word2vec import ClasificadorWord2Vec

layout = dbc.Container([
    dcc.Store(id='store-bert', storage_type='session'),
    dcc.Store(id='store-w2v', storage_type='session'),

    html.H2("Clasificación de Género", className="text-info mb-2"),
    dbc.Button("Ejecutar Bert", id="btn-bert", color="danger", size="sm", className="mb-4"),
dbc.Button("Ejecutar Word2Vec", id="btn-w2v", color="primary", size="sm", className="mb-4"),

    dcc.Loading(type="circle", children=[
        html.Div(id="bert-display-area"),
        html.Div(id="w2v-display-area")
        # Aquí metemos lo que se debe quedar fijo
    ])
], fluid=True)


# ENTRENAMIENTO
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


# CALLBACK 2: VISUALIZACIÓN PERSISTENTE
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

#-----------------------------
#callback w2v
#calculo w2v
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

#visualizacion w2v
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

