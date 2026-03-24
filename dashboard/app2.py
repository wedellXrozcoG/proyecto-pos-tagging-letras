#ejecutable
import dash
import dash_bootstrap_components as dbc
from dash import html, Input, Output
from interfaz import pg_clasificacion, pg_clustering, pg_tsne

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    assets_folder="assets", # Mantenemos la carpeta
    suppress_callback_exceptions=True
)

# usar solo style2.css
#inicio estilos
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>NLP Dashboard</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="/assets/style2.css"> 
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = dbc.Container([
    # Barra de título negra
    dbc.Navbar(
        dbc.Container(
            dbc.Row(
                dbc.Col(html.H4("NLP Dashboard", className="text-white mb-0"), width="auto"),
                justify="center",
                style={"width": "100%"}
            )
        ),
        color="black", #inicio estilo
        dark=True,
    ),

    dbc.Tabs([
        dbc.Tab(label="Inicio",          tab_id="tab-home"),
        dbc.Tab(label="Clasificación",   tab_id="tab-cls"),
        dbc.Tab(label="Clustering",      tab_id="tab-clustering"),
        dbc.Tab(label="t-SNE",           tab_id="tab-tsne"),
        dbc.Tab(label="Búsqueda Semántica", tab_id="bus-sem")
    ],
    id="tabs",
    active_tab="tab-home",
    style={"backgroundColor": "black", "border": "none"},
    className="nav-fill"
    ),

    # Área de contenido blanca
    html.Div(
        id="contenido",
        className="p-4",
        style={
            "backgroundColor": "white",
            "minHeight": "100vh",
            "color": "black"
        }
    )
], fluid=True, style={"padding": "0"}) #fin estilos

@app.callback(
    Output("contenido", "children"),
    Input("tabs", "active_tab")
)
def render(tab):
    if tab == "tab-home":
        return html.H4("😶‍🌫️ Seleccione una pestaña",
                        className="text-muted mt-4")
    elif tab == "tab-cls":
        return pg_clasificacion.layout
    elif tab == "tab-clustering":
        return pg_clustering.layout
    elif tab == "tab-tsne":
        return pg_tsne.layout
    elif tab == "bus-sem":
        return "proximamenteee"


if __name__ == "__main__":
    app.run(debug=True, port=8050)