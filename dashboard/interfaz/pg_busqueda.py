import dash
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
import gc
from src.db_manager import get_collection
from src.analisis_nlp.Bert.analisis_semantico import BuscadorSemantico

layout = dbc.Container([
    # El Store guarda solo TEXTO, no ocupa casi nada de RAM
    dcc.Store(id='store-busqueda-semantica', storage_type='session'),

    html.H2("Semantic Search (English)", className="text-info mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Input(id="input-busqueda", placeholder="Search meaning...", type="text"),
        ], width=9),
        dbc.Col([
            dbc.Button("Search", id="btn-buscar", color="danger", className="w-100"),
        ], width=3),
    ], className="mb-4"),

    dcc.Loading(type="circle", children=[
        html.Div(id="resultados-busqueda-display")
    ])
], fluid=True)


# CALLBACK 1: CALCULAR (Carga el modelo, busca y lo borra de la RAM)
@callback(
    Output("store-busqueda-semantica", "data"),
    Input("btn-buscar", "n_clicks"),
    State("input-busqueda", "value"),
    prevent_initial_call=True
)
def buscar_y_guardar(n_clicks, texto):
    if not texto: return None

    col = get_collection()
    # Aquí entra el modelo a la RAM (1.5GB aprox)
    buscador = BuscadorSemantico(col)
    resultados = buscador.buscar_similares(texto)

    # Aquí sacamos el modelo de la RAM
    del buscador
    gc.collect()

    return resultados


# CALLBACK 2: MOSTRAR (Solo dibuja lo que está en el Store)
@callback(
    Output("resultados-busqueda-display", "children"),
    Input("store-busqueda-semantica", "data")
)
def renderizar_resultados(data):
    if not data:
        return html.Div("Search for something to see results.", className="text-muted")

    return dash_table.DataTable(
        data=data,
        columns=[{"name": i, "id": i} for i in ["Titulo", "Artista", "Género", "Similitud"]],
        style_header={'backgroundColor': 'black', 'color': 'white'},
        style_cell={'textAlign': 'left', 'padding': '10px', 'color': 'black'}
    )