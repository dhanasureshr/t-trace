"""Configuration panel UI components for M-TRACE AnalysisEngine."""
from dash import html, dcc
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigUI:
    """Configuration panel UI components."""
    
    @staticmethod
    def create_config_layout(config_path: Path = None) -> html.Div:
        """
        Create configuration panel layout.
        
        Args:
            config_path: Path to config.yml file (optional)
        
        Returns:
            Dash HTML layout for configuration panel
        """
        # Load current config if available
        current_config = {}
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    current_config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        return html.Div([
            html.H2("M-TRACE Configuration", style={"marginBottom": "20px"}),
            
            # Mode Selection
            html.Div([
                html.H4("Logging Mode", style={"marginTop": "20px"}),
                dcc.RadioItems(
                    id="config-mode",
                    options=[
                        {"label": "Development (Detailed Logging)", "value": "development"},
                        {"label": "Production (Lightweight Logging)", "value": "production"}
                    ],
                    value=current_config.get("mode", "development"),
                    labelStyle={"display": "block", "marginBottom": "10px"}
                )
            ]),
            
            # Sparse Logging Configuration
            html.Div([
                html.H4("Sparse Logging", style={"marginTop": "30px"}),
                dcc.Checklist(
                    id="sparse-logging-enabled",
                    options=[{"label": "Enable Sparse Logging", "value": "enabled"}],
                    value=["enabled"] if current_config.get("sparse_logging", {}).get("enabled", True) else [],
                    style={"marginBottom": "15px"}
                ),
                html.Div([
                    html.Label("Sparse Threshold:", style={"fontWeight": "bold"}),
                    dcc.Input(
                        id="sparse-threshold",
                        type="number",
                        value=current_config.get("sparse_logging", {}).get("sparse_threshold", 0.1),
                        min=0, max=1, step=0.01,
                        style={"width": "100px", "marginLeft": "10px"}
                    ),
                    html.Label(" (Log values with abs(value) > threshold)", style={"marginLeft": "10px"})
                ], style={"marginBottom": "10px"}),
                html.Div([
                    html.Label("Top-K Values:", style={"fontWeight": "bold"}),
                    dcc.Input(
                        id="sparse-top-k",
                        type="number",
                        value=current_config.get("sparse_logging", {}).get("top_k_values", 5),
                        min=1, max=100, step=1,
                        style={"width": "100px", "marginLeft": "10px"}
                    ),
                    html.Label(" (Always log top-k values)", style={"marginLeft": "10px"})
                ])
            ]),
            
            # Compression Configuration
            html.Div([
                html.H4("Compression", style={"marginTop": "30px"}),
                dcc.Checklist(
                    id="compression-enabled",
                    options=[{"label": "Enable Compression", "value": "enabled"}],
                    value=["enabled"] if current_config.get("compression", {}).get("enabled", True) else [],
                    style={"marginBottom": "15px"}
                ),
                html.Div([
                    html.Label("Algorithm:", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="compression-algorithm",
                        options=[
                            {"label": "Snappy (Fast)", "value": "snappy"},
                            {"label": "Zstandard (Balanced)", "value": "zstd"},
                            {"label": "Gzip (High Compression)", "value": "gzip"},
                            {"label": "None", "value": "none"}
                        ],
                        value=current_config.get("compression", {}).get("compression_type", "snappy"),
                        style={"width": "300px", "marginLeft": "10px"}
                    )
                ], style={"marginBottom": "10px"}),
                html.Div([
                    html.Label("Compression Level:", style={"fontWeight": "bold"}),
                    dcc.Slider(
                        id="compression-level",
                        min=1, max=9, step=1,
                        value=current_config.get("compression", {}).get("compression_level", 1),
                        marks={i: str(i) for i in range(1, 10)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Small("1 = Fastest, 9 = Maximum Compression", style={"marginLeft": "10px"})
                ])
            ]),
            
            # Logging Frequency
            html.Div([
                html.H4("Logging Frequency", style={"marginTop": "30px"}),
                html.Div([
                    html.Label("Batch Size:", style={"fontWeight": "bold"}),
                    dcc.Input(
                        id="logging-batch-size",
                        type="number",
                        value=current_config.get("logging_frequency", {}).get("batch_size", 1000),
                        min=1, max=10000, step=100,
                        style={"width": "120px", "marginLeft": "10px"}
                    ),
                    html.Label(" logs", style={"marginLeft": "5px"})
                ], style={"marginBottom": "10px"}),
                html.Div([
                    html.Label("Time Interval:", style={"fontWeight": "bold"}),
                    dcc.Input(
                        id="logging-time-interval",
                        type="number",
                        value=current_config.get("logging_frequency", {}).get("time_interval", 60),
                        min=1, max=3600, step=10,
                        style={"width": "120px", "marginLeft": "10px"}
                    ),
                    html.Label(" seconds", style={"marginLeft": "5px"})
                ])
            ]),
            
            # Custom Fields Selection
            html.Div([
                html.H4("Custom Fields to Log", style={"marginTop": "30px"}),
                html.P("Select additional fields to log (default fields always logged):", 
                       style={"marginBottom": "10px"}),
                dcc.Checklist(
                    id="custom-fields",
                    options=[
                        {"label": "Attention Weights", "value": "attention_weights"},
                        {"label": "Feature Maps", "value": "feature_maps"},
                        {"label": "Gradients", "value": "gradients"},
                        {"label": "Feature Importance", "value": "feature_importance"},
                        {"label": "Layer Activations", "value": "layer_activations"},
                        {"label": "Uncertainty Estimates", "value": "uncertainty_estimates"},
                        {"label": "Training Dynamics", "value": "training_dynamics"},
                        {"label": "Input/Output Data", "value": "contextual_info"}
                    ],
                    value=current_config.get("custom_fields", [
                        "attention_weights", "feature_maps", "losses"
                    ]),
                    labelStyle={"display": "block", "marginBottom": "5px"}
                )
            ]),
            
            # Action Buttons
            html.Div([
                html.Button(
                    "Save Configuration",
                    id="save-config-btn",
                    n_clicks=0,
                    style={
                        "marginTop": "30px",
                        "padding": "10px 20px",
                        "backgroundColor": "#4CAF50",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "5px",
                        "cursor": "pointer"
                    }
                ),
                html.Button(
                    "Reset to Defaults",
                    id="reset-config-btn",
                    n_clicks=0,
                    style={
                        "marginTop": "30px",
                        "marginLeft": "10px",
                        "padding": "10px 20px",
                        "backgroundColor": "#f44336",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "5px",
                        "cursor": "pointer"
                    }
                ),
                html.Div(id="config-save-status", style={"marginTop": "10px", "color": "green"})
            ])
        ], style={"padding": "20px", "maxWidth": "800px", "margin": "0 auto"})
    
    @staticmethod
    def save_config(
        config_path: Path,
        mode: str,
        sparse_enabled: bool,
        sparse_threshold: float,
        sparse_top_k: int,
        compression_enabled: bool,
        compression_algorithm: str,
        compression_level: int,
        batch_size: int,
        time_interval: int,
        custom_fields: list
    ) -> bool:
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save config.yml
            mode: "development" or "production"
            sparse_enabled: Whether sparse logging is enabled
            sparse_threshold: Sparse logging threshold
            sparse_top_k: Top-k values to log
            compression_enabled: Whether compression is enabled
            compression_algorithm: Compression algorithm
            compression_level: Compression level (1-9)
            batch_size: Logging batch size
            time_interval: Time interval in seconds
            custom_fields: List of custom fields to log
        
        Returns:
            True if save succeeded
        """
        try:
            config = {
                "mode": mode,
                "sparse_logging": {
                    "enabled": sparse_enabled,
                    "sparse_threshold": sparse_threshold,
                    "top_k_values": sparse_top_k
                },
                "compression": {
                    "enabled": compression_enabled,
                    "compression_type": compression_algorithm,
                    "compression_level": compression_level
                },
                "logging_frequency": {
                    "batch_size": batch_size,
                    "time_interval": time_interval
                },
                "custom_fields": custom_fields,
                "default_fields": [
                    "model_type", "framework", "timestamp", "run_id", 
                    "mode", "layer_name", "losses"
                ]
            }
            
            # Ensure parent directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to YAML
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False