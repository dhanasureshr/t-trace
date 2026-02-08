import dash
from dash import dcc, html
from dash.dependencies import Input, Output,State
import plotly.express as px

# Initialize Dash app
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True
    )

# Define layout
app.layout = html.Div([
    html.H1("M-TRACE Dashboard"),
    dcc.Tabs(id="tabs", value="logging", children=[
        dcc.Tab(label="Logging", value="logging"),
        dcc.Tab(label="Analysis", value="analysis"),
        dcc.Tab(label="Configuration", value="configuration"),
    ]),
    html.Div(id="tab-content"),
   
])




# Callback to update tab content
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
)
def render_tab(tab):
    if tab == "logging":
        return html.Div([
            html.H3("Real-Time Logs"),
            dcc.Textarea(id="logs", style={"width": "100%", "height": 300}),
            dcc.Interval(id="interval", interval=1000, n_intervals=0)
        ])
    elif tab == "analysis":
        return html.Div([
            html.H3("Attention Heatmap"),
            dcc.Graph(id="heatmap")
        ])
    elif tab == "configuration":
        return html.Div([
            html.H3("Configuration"),
            dcc.Dropdown(id="log-level", options=[
                {"label": "Minimal", "value": "minimal"},
                {"label": "Detailed", "value": "detailed"},
                {"label": "Debug", "value": "debug"}
            ], value="detailed")
        ])

# Run app
if __name__ == "__main__":
    app.run(port=8060, debug=True)