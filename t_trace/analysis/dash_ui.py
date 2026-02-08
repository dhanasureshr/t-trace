import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from transformers import BertTokenizer

# Load the Parquet file
logs_df = pd.read_parquet("/home/dhana/Documents/Ai/mtrace/t-trace/logs/bert_layer_logs.parquet")

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Sample input text
text = "I hate my job; it's stressful, unfulfilling, and draining."

# Tokenize the input text
tokens = tokenizer.tokenize(text)  # Get token strings
token_ids = tokenizer(text, return_tensors='pt')["input_ids"]  # Get token IDs

# Add special tokens to the tokens list
tokens = ["[CLS]"] + tokens + ["[SEP]"]

# Extract attention weights
attention_weights = logs_df["data"].apply(lambda x: x["attention_weights"])

def generate_heatmap(tokens, attention_weights):
    """
    Generate a heatmap of attention weights using Plotly.
    
    Args:
        tokens (list): List of input tokens.
        attention_weights (list or np.ndarray): Attention weights for each token.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    # Convert attention_weights to a NumPy array
    attention_weights = np.array(attention_weights)
    
    # Compute the average attention weights across all layers
    average_attention_weights = np.mean(attention_weights, axis=0)
    
    # Ensure the number of tokens matches the length of attention weights
    max_seq_length = len(average_attention_weights)  # Maximum sequence length (e.g., 256)

    # Pad or truncate tokens to match the length of attention weights
    if len(tokens) < max_seq_length:
        tokens += ["[PAD]"] * (max_seq_length - len(tokens))
    elif len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]

    # Remove special tokens from tokens and attention_weights
    special_tokens = ["[CLS]", "[SEP]", "[PAD]"]
    filtered_tokens = [token for token in tokens if token not in special_tokens]
    filtered_weights = [weight for token, weight in zip(tokens, average_attention_weights) if token not in special_tokens]

    # Normalize attention weights to [0, 1] for color intensity
    normalized_weights = [float(w) / max(filtered_weights) for w in filtered_weights]

    # Remove stop words
    stop_words = ["the", "is", "and", "a", "an", "in", "it", "of", "to", ".", ","]
    filtered_tokens = [token for token in filtered_tokens if token not in stop_words]
    filtered_weights = [weight for token, weight in zip(filtered_tokens, normalized_weights) if token not in stop_words]

    # Create a heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=[filtered_weights],
        x=filtered_tokens,
        y=["Attention Weights"],
        colorscale="Reds",
        colorbar=dict(title="Attention Weight")
    ))

    # Update layout for better visualization
    fig.update_layout(
        title="Attention Heatmap",
        xaxis=dict(title="Tokens"),
        yaxis=dict(title=""),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

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

# Callback to update the heatmap
@app.callback(
    Output("heatmap", "figure"),
    Input("tabs", "value"),
)
def update_heatmap(tab):
    if tab == "analysis":
        return generate_heatmap(tokens, attention_weights)
    return go.Figure()

# Run app
if __name__ == "__main__":
    app.run(host="192.168.1.35",port=8060, debug=True)