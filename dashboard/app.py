import dash
import pandas as pd
from dash import dcc, Dash, callback, Input, Output, ctx
from dash import html

#llamado de clases
from src.analysis.Analisismorfologico import AnalisisMorfologico
from src.analysis.temporal_analyzer import TemporalAnalyzer
from src.data.preprocessorEDA import preprocesador

# Cargar análisis

df = pd.read_csv('../data/processed/spotify_clean02.csv', sep=';')

analisis = AnalisisMorfologico(df)
analisis.procesar_corpus()
temporal = TemporalAnalyzer(df)
eda = preprocesador(df)

app = Dash(__name__)

app.layout = html.Div(className='main-container', children=[

    html.Div(className='header', children=[
        html.H1('Análisis Morfosintáctico de Letras Musicales'),
        html.H5('Análisis comparativo Hip-Hop vs Pop'),
    ]),

    html.Div(className='nav-container', children=[
        html.Button('Distribución POS', id='btn-nclicks-1', n_clicks=0, className='nav-button'),
        html.Button('Evolución Temporal', id='btn-nclicks-2', n_clicks=0, className='nav-button'),
        html.Button('Comparación de Género', id='btn-nclicks-3', n_clicks=0, className='nav-button'),
        html.Button('Emocionalidad', id='btn-nclicks-4', n_clicks=0, className='nav-button'),
    ]),

    html.Div(id='container-button-timestamp', className='content-area')

])

# CALLBACK PRINCIPAL
@callback(
    Output('container-button-timestamp', 'children'),
    Input('btn-nclicks-1', 'n_clicks'),
    Input('btn-nclicks-2', 'n_clicks'),
    Input('btn-nclicks-3', 'n_clicks'),
    Input('btn-nclicks-4', 'n_clicks')
)
def displayClick(btn1, btn2, btn3, btn4):

    if ctx.triggered_id == "btn-nclicks-1":
        return html.Div([
            html.H3("Distribución POS Tagging"),
            dcc.Tabs(id="tabs-pos", value='tab-pos-1', children=[
                dcc.Tab(label='Distribución Categorías POS', value='tab-pos-1'),
                dcc.Tab(label='Métricas Derivadas', value='tab-pos-2'),
                dcc.Tab(label='Patrones por Género', value='tab-pos-3'),
            ]),
            html.Div(id='tabs-content-pos')
        ])

    elif ctx.triggered_id == "btn-nclicks-2":
        return html.Div([
            html.H3("Evolución Temporal"),
            dcc.Tabs(id="tabs-tiempo", value='tab-tiempo-1', children=[
                dcc.Tab(label='Palabras por década', value='tab-tiempo-1'),
                dcc.Tab(label='Emociones por década', value='tab-tiempo-2'),
                dcc.Tab(label='Popularidad', value='tab-tiempo-3'),
            ]),
            html.Div(id='tabs-content-tiempo')
        ])

    elif ctx.triggered_id == "btn-nclicks-3":
        return html.Div([
            html.H3("Comparación de Género"),
            dcc.Tabs(id="tabs-genero", value='tab-genero-1', children=[
                dcc.Tab(label='Géneros con más pronombres', value='tab-genero-1'),
                dcc.Tab(label='Popularidad de género', value='tab-genero-2'),
            ]),
            html.Div(id='tabs-content-genero')
        ])

    elif ctx.triggered_id == "btn-nclicks-4":
        return html.Div([
            html.H3("Emocionalidad"),
            dcc.Tabs(id="tabs-emociones", value='tab-emo-1', children=[
                dcc.Tab(label='Hip-Hop', value='tab-emo-1'),
                dcc.Tab(label='Pop', value='tab-emo-2'),
            ]),
            html.Div(id='tabs-content-emociones')
        ])

    return html.Div("Selecciona un botón para ver el análisis")


# CALLBACK POS
@callback(
    Output('tabs-content-pos', 'children'),
    Input('tabs-pos', 'value')
)
def render_pos_tabs(tab):
    if tab == 'tab-pos-1':
        return dcc.Graph(figure=analisis.distribucion_pos_completa())
    elif tab == 'tab-pos-2':
        return dcc.Graph(figure=analisis.calcular_metricas_derivadas())
    elif tab == 'tab-pos-3':
        return dcc.Graph(figure=analisis.patrones_por_genero())


# CALLBACK TIEMPO
@callback(
    Output('tabs-content-tiempo', 'children'),
    Input('tabs-tiempo', 'value')
)
def render_tiempo_tabs(tab):
    if tab == 'tab-tiempo-1':
        return html.Div([
            html.H4("¿Cuáles palabras se usaron más por década?"),
            dcc.Graph(
                figure=temporal.palabras_mas_usadas_por_decada()
            )
        ])
    elif tab == 'tab-tiempo-2':
        return html.Div([
            html.H4("¿Cómo se distribuyeron las emociones por década?"),
            dcc.Graph(
                figure=temporal.visualizar_sentimientos()
            )
        ])
    elif tab == 'tab-tiempo-3':
        return html.Div([
            html.H4("¿Cómo evolucionó la popularidad en el tiempo?"),
            dcc.Graph(
                figure=eda.grafico_popularidad_tiempo()
            )
        ])


# CALLBACK GENERO
@callback(
    Output('tabs-content-genero', 'children'),
    Input('tabs-genero', 'value')
)
def render_genero_tabs(tab):
    if tab == 'tab-genero-1':
        return html.Div([
            html.H4("¿Qué géneros usan más pronombres (1ra, 2da, 3ra persona)?"),

            dcc.Graph(
                figure=analisis.analisis_pronombres()
            )
        ])
    elif tab == 'tab-genero-2':
        return html.Div([
            html.H4("¿Cómo fue la popularidad según el género?"),

            dcc.Graph(
                figure=eda.grafico_popularidad_genero()
            )
        ])



# CALLBACK EMOCIONES
@callback(
    Output('tabs-content-emociones', 'children'),
    Input('tabs-emociones', 'value')
)
def render_emociones_tabs(tab):
    if tab == 'tab-emo-1':
        return html.Div([
            html.H4("¿Qué emoción predominaba más en el hip-hop / pop? ¿Existe alguna diferencia relevante?"),

            dcc.Graph(
                figure=eda.grafico_pie('hip hop')
            )
        ])
    elif tab == 'tab-emo-2':
        return html.Div([
            html.H4("¿Qué emoción predominaba más en el hip-hop / pop? ¿Existe alguna diferencia relevante?"),

            dcc.Graph(
                figure=eda.grafico_pie('pop')
            )
        ])

app.run(debug=False)
