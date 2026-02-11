"""Main M-TRACE Analysis Dashboard implementation."""
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL

from .data_loader import DataLoader
from .visualizations import Visualizations
from .config_ui import ConfigUI
from .callbacks import register_callbacks

logger = logging.getLogger(__name__)


class AnalysisDashboard:
    """
    Main M-TRACE Analysis Dashboard class.
    
    Provides interactive visualizations and configuration UI for M-TRACE logs
    across all frameworks (PyTorch, TensorFlow, scikit-learn).
    
    Implements Section 2.3 of M-TRACE specification:
    - Home panel: Overview of logged runs
    - Analysis panel: Interactive visualizations (attention heatmaps, loss curves, etc.)
    - Configuration panel: YAML config editing through UI
    """
    
    def __init__(self, storage_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Analysis Dashboard.
        
        Args:
            storage_config: Optional configuration for StorageEngine integration
        """
        self.storage_config = storage_config or {}
        self.config_path = Path(self.storage_config.get("config_path", "config.yml"))
        
        # Initialize components
        self.data_loader = DataLoader(storage_config=self.storage_config)
        self.visualizations = Visualizations()
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.app.title = "M-TRACE Analysis Dashboard"
        
        # Set up layout
        self._setup_layout()
        
        # Register callbacks
        register_callbacks(self.app, self.data_loader, self.visualizations, self.config_path)
        
        logger.info("M-TRACE Analysis Dashboard initialized")
    
    def _setup_layout(self) -> None:
        """Set up the main dashboard layout with tabs."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("M-TRACE: Model Transparency through Recursive Analysis of Contextual Encapsulation",
                       style={"textAlign": "center", "marginBottom": "5px"}),
                html.P("Interactive Dashboard for Model Interpretability and Analysis",
                       style={"textAlign": "center", "color": "#666", "marginBottom": "20px"})
            ], style={"backgroundColor": "#f8f9fa", "padding": "15px", "marginBottom": "20px"}),
            
            # Tabs
            dcc.Tabs(id="main-tabs", value="home", children=[
                dcc.Tab(label="🏠 Home", value="home", style={"fontWeight": "bold"}),
                dcc.Tab(label="📊 Analysis", value="analysis", style={"fontWeight": "bold"}),
                dcc.Tab(label="⚙️ Configuration", value="configuration", style={"fontWeight": "bold"}),
            ], style={"marginBottom": "20px"}),
            
            # Tab Content
            html.Div(id="tab-content"),
            
            # Store components for client-side state
            dcc.Store(id="selected-run-id", data=None),
            dcc.Store(id="refresh-trigger", data=0),
        ], style={"maxWidth": "1400px", "margin": "0 auto", "padding": "10px"})
    
    def run(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False) -> None:
        """
        Start the Dash dashboard server.
        
        Args:
            host: Host address to bind (default: "127.0.0.1")
            port: Port number (default: 8050)
            debug: Enable debug mode with hot reloading (default: False)
        """
        logger.info(f"Starting M-TRACE Dashboard at http://{host}:{port}")
        logger.info("Press Ctrl+C to stop the server")
        
        try:
            self.app.run(
                host=host,
                port=port,
                debug=debug,
                use_reloader=debug
            )
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            raise