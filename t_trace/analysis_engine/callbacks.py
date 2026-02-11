"""Dash callbacks for M-TRACE AnalysisEngine."""
import logging
import numpy as np  # ← CRITICAL ADDITION: Required for sparse reconstruction
from pathlib import Path
from typing import Any, Dict, List, Optional
import dash
import pandas as pd
from dash import html, dcc, no_update, ctx
from dash.dependencies import Input, Output, State, MATCH, ALL
import plotly.graph_objects as go

import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )
)
from t_trace.analysis_engine.visualizations import _is_empty_array
from .data_loader import DataLoader
from .visualizations import Visualizations
from .config_ui import ConfigUI

logger = logging.getLogger(__name__)


def register_callbacks(
    app,
    data_loader: DataLoader,
    visualizations: Visualizations,
    config_path: Path
):
    """Register all Dash callbacks for the dashboard."""
    
    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
        State("selected-run-id", "data"),
        prevent_initial_call=False
    )
    def render_tab_content(tab: str, selected_run_id: str) -> Any:
        """Render content for selected tab with debug logging."""
        logger.info(f"render_tab_content called: tab={tab}, selected_run_id={selected_run_id}")
        
        if tab == "home":
            content = _render_home_panel(data_loader)
            logger.info(f"Home panel rendered with content type: {type(content)}")
            return content
        elif tab == "analysis":
            content = _render_analysis_panel(selected_run_id, data_loader)
            logger.info(f"Analysis panel rendered with content type: {type(content)}")
            return content
        elif tab == "configuration":
            content = ConfigUI.create_config_layout(config_path=config_path)
            logger.info(f"Configuration panel rendered")
            return content
        return html.Div("Unknown tab")
    
    @app.callback(
        Output("selected-run-id", "data"),
        Input({"type": "run-row", "index": ALL}, "n_clicks"),
        State({"type": "run-row", "index": ALL}, "id"),
        prevent_initial_call=True
    )
    def select_run(n_clicks_list: List[int], ids: List[Dict]) -> Optional[str]:
        """Handle run selection from home panel."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return None
        
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        triggered_dict = eval(triggered_id)
        
        logger.info(f"Run selected: {triggered_dict['index']}")
        return triggered_dict["index"]
    
    @app.callback(
        Output("config-save-status", "children"),
        Input("save-config-btn", "n_clicks"),
        Input("reset-config-btn", "n_clicks"),
        State("config-mode", "value"),
        State("sparse-logging-enabled", "value"),
        State("sparse-threshold", "value"),
        State("sparse-top-k", "value"),
        State("compression-enabled", "value"),
        State("compression-algorithm", "value"),
        State("compression-level", "value"),
        State("logging-batch-size", "value"),
        State("logging-time-interval", "value"),
        State("custom-fields", "value"),
        prevent_initial_call=True
    )
    def handle_config_actions(
        save_clicks: int,
        reset_clicks: int,
        mode: str,
        sparse_enabled: List[str],
        sparse_threshold: float,
        sparse_top_k: int,
        compression_enabled: List[str],
        compression_algorithm: str,
        compression_level: int,
        batch_size: int,
        time_interval: int,
        custom_fields: List[str]
    ) -> str:
        """Handle both save and reset config actions in single callback."""
        triggered_id = ctx.triggered_id
        
        if triggered_id == "save-config-btn":
            logger.info("Save configuration button clicked")
            success = ConfigUI.save_config(
                config_path=config_path,
                mode=mode,
                sparse_enabled=bool(sparse_enabled),
                sparse_threshold=sparse_threshold,
                sparse_top_k=sparse_top_k,
                compression_enabled=bool(compression_enabled),
                compression_algorithm=compression_algorithm,
                compression_level=compression_level,
                batch_size=batch_size,
                time_interval=time_interval,
                custom_fields=custom_fields
            )
            
            if success:
                return html.Div([
                    html.Span("✓ ", style={"color": "green", "fontSize": "20px"}),
                    "Configuration saved successfully!"
                ])
            else:
                return html.Div([
                    html.Span("✗ ", style={"color": "red", "fontSize": "20px"}),
                    "Error saving configuration. Check logs for details."
                ])
        
        elif triggered_id == "reset-config-btn":
            logger.info("Reset configuration button clicked")
            default_config = {
                "mode": "development",
                "sparse_logging": {
                    "enabled": True,
                    "sparse_threshold": 0.1,
                    "top_k_values": 5
                },
                "compression": {
                    "enabled": True,
                    "compression_type": "snappy",
                    "compression_level": 1
                },
                "logging_frequency": {
                    "batch_size": 1000,
                    "time_interval": 60
                },
                "custom_fields": ["attention_weights", "feature_maps", "losses"]
            }
            
            try:
                import yaml
                with open(config_path, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
                return html.Div([
                    html.Span("✓ ", style={"color": "green", "fontSize": "20px"}),
                    "Configuration reset to defaults!"
                ])
            except Exception as e:
                logger.error(f"Error resetting configuration: {e}")
                return html.Div([
                    html.Span("✗ ", style={"color": "red", "fontSize": "20px"}),
                    "Error resetting configuration."
                ])
        
        return no_update


def _render_home_panel(data_loader: DataLoader) -> html.Div:
    """Render home panel with run overview and comprehensive debug logging."""
    try:
        logger.info("Rendering home panel - calling list_runs()")
        
        try:
            runs = data_loader.list_runs()
            logger.info(f"list_runs() returned {len(runs)} runs")
        except Exception as e:
            logger.error(f"Error in list_runs(): {e}", exc_info=True)
            runs = []
        
        if not runs:
            logger.warning("No runs found - displaying empty state")
            return html.Div([
                html.H2("No Runs Found", style={"textAlign": "center", "marginTop": "50px"}),
                html.P(
                    "Enable logging with enable_logging(model, mode='development') to start capturing logs.",
                    style={"textAlign": "center", "color": "#666"}
                ),
                html.Div([
                    html.P(f"Storage directory: {data_loader.storage_dir}", 
                           style={"textAlign": "center", "fontSize": "12px", "color": "#999"}),
                    html.P(f"Development dir exists: {(data_loader.storage_dir / 'development').exists()}",
                           style={"textAlign": "center", "fontSize": "12px", "color": "#999"}),
                    html.P(f"Production dir exists: {(data_loader.storage_dir / 'production').exists()}",
                           style={"textAlign": "center", "fontSize": "12px", "color": "#999"}),
                ], style={"marginTop": "20px"})
            ], style={"padding": "20px"})
        
        logger.info(f"Creating table with {len(runs)} runs")
        run_rows = []
        for run in runs[:20]:
            summary = {}
            try:
                summary = data_loader.get_run_summary(run["run_id"])
            except Exception as e:
                logger.debug(f"Error getting summary for {run['run_id']}: {e}")
            
            run_rows.append(html.Tr([
                html.Td(run["run_id"][:8]),
                html.Td(run["model_type"]),
                html.Td(run["mode"]),
                html.Td(run["timestamp"]),
                html.Td(f"{run.get('size_bytes', 0) / 1024:.1f} KB"),
                html.Td(summary.get("total_logs", "N/A")),
                html.Td(html.Button(
                    "Select",
                    id={"type": "run-row", "index": run["run_id"]},
                    n_clicks=0,
                    style={"padding": "5px 10px", "backgroundColor": "#007bff", "color": "white", 
                           "border": "none", "borderRadius": "3px", "cursor": "pointer"}
                ))
            ]))
        
        logger.info(f"Home panel rendering complete with {len(run_rows)} rows")
        return html.Div([
            html.H2("Logged Runs Overview"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Run ID"), html.Th("Model Type"), html.Th("Mode"), 
                    html.Th("Timestamp"), html.Th("Size"), html.Th("Logs"), html.Th("Action")
                ])),
                html.Tbody(run_rows)
            ], style={"width": "100%", "borderCollapse": "collapse", "marginTop": "20px"}),
            html.P(f"Showing {len(runs[:20])} of {len(runs)} runs", style={"marginTop": "10px", "color": "#666"})
        ], style={"padding": "20px"})
        
    except Exception as e:
        logger.error(f"Error rendering home panel: {e}", exc_info=True)
        return html.Div(f"Error loading runs: {str(e)}", style={"padding": "20px", "color": "red"})


def _render_analysis_panel(selected_run_id: str, data_loader: DataLoader) -> html.Div:
    """Render analysis panel with visualizations."""
    if not selected_run_id:
        return html.Div([
            html.H2("Analysis Panel"),
            html.P("Select a run from the Home panel to view analysis.", 
                   style={"marginTop": "50px", "textAlign": "center", "color": "#666"})
        ], style={"padding": "20px"})
    
    try:
        logger.info(f"Loading logs for run: {selected_run_id}")
        
        df = data_loader.load_run_logs(selected_run_id)
        if df is None or df.empty:
            logger.warning(f"No logs found for run {selected_run_id}")
            return html.Div([
                html.H2(f"Analysis: Run {selected_run_id[:8]}"),
                html.Div([
                    html.P("⚠️ No log data available for analysis", 
                        style={"color": "#f44336", "fontSize": "18px", "textAlign": "center"}),
                    html.P("Possible causes:", style={"marginTop": "10px", "fontWeight": "bold"}),
                    html.Ul([
                        html.Li("Run was executed in production mode (gradients not logged)"),
                        html.Li("Model didn't execute forward/backward passes during logging"),
                        html.Li("Log file corrupted or schema mismatch")
                    ], style={"marginLeft": "20px"}),
                    html.P(f"Storage path: {data_loader.storage_dir}", 
                        style={"fontSize": "12px", "color": "#999", "marginTop": "20px"})
                ], style={"padding": "30px", "textAlign": "center"})
            ], style={"padding": "20px"})
        
        logger.info(f"Loaded {len(df)} logs for analysis")
        summary = data_loader.get_run_summary(selected_run_id)
        viz = Visualizations()
        
        # Create core visualizations
        loss_fig = viz.create_loss_curve(df)
        activations_fig = viz.create_layer_activations_chart(df)
        gradient_fig = viz.create_gradient_norm_chart(df)
        
        # === CRITICAL FIX: Robust attention & feature importance extraction ===
        attention_fig = None
        feature_importance_fig = None

                # Add this right before the for-loop in _render_analysis_panel()
        if df is not None and len(df) > 0:
            sample_row = df.iloc[0]
            if "internal_states" in sample_row and isinstance(sample_row["internal_states"], dict):
                attn_sample = sample_row["internal_states"].get("attention_weights")
                logger.info(f"DEBUG: Attention data type={type(attn_sample)}, keys={list(attn_sample.keys()) if isinstance(attn_sample, dict) else 'N/A'}")

        for _, row in df.iterrows():
            internal_states = row.get("internal_states")
            if not isinstance(internal_states, dict):
                continue
            
            # Process attention weights (handle BOTH sparse dict AND raw list)
            if attention_fig is None:
                attn_data = internal_states.get("attention_weights")
                if attn_data is not None:
                    try:
                        # CASE 1: Sparse representation (dict with sparse metadata)
                        if isinstance(attn_data, dict):
                            logger.debug(f"Processing sparse attention data: keys={list(attn_data.keys())}")
                            sparse_vals = attn_data.get("sparse_values", [])
                            sparse_idxs = attn_data.get("sparse_indices", [])
                            shape = attn_data.get("shape", [1, 1])
                            
                            # Reconstruct dense array
                            dense = np.zeros(np.prod(shape), dtype=np.float32)
                            if sparse_idxs and sparse_vals:
                                dense[sparse_idxs] = sparse_vals
                            dense = dense.reshape(shape)
                            
                            # Extract 2D matrix for visualization (BERT: [batch, heads, seq, seq])
                            if dense.ndim == 4:
                                viz_data = dense[0, 0]  # First batch, first head
                            elif dense.ndim == 3:
                                viz_data = dense[0]  # First head
                            else:
                                viz_data = dense
                            
                            if viz_data.ndim == 2:
                                attention_fig = viz.create_attention_heatmap(
                                    attention_weights=viz_data.tolist(),
                                    layer_name=internal_states.get("layer_name", "Layer"),
                                    head_idx=0
                                )
                                logger.info(f"✓ Created attention heatmap from sparse data (shape: {viz_data.shape})")
                            else:
                                logger.warning(f"Reconstructed attention is not 2D (shape: {viz_data.shape})")
                        
                        # CASE 2: Raw array/list (non-sparse logging OR small tensors)
                        elif isinstance(attn_data, (list, np.ndarray)) and len(attn_data) > 0:
                            logger.debug(f"Processing raw attention data: type={type(attn_data)}, len={len(attn_data)}")
                            # Convert to list if numpy array
                            attn_list = attn_data.tolist() if isinstance(attn_data, np.ndarray) else attn_data
                            
                            # Reshape if flattened (assume square matrix)
                            if len(attn_list) > 0 and isinstance(attn_list[0], (int, float)):
                                seq_len = int(np.sqrt(len(attn_list)))
                                if seq_len * seq_len == len(attn_list):
                                    attn_2d = np.array(attn_list).reshape(seq_len, seq_len)
                                    attention_fig = viz.create_attention_heatmap(
                                        attention_weights=attn_2d.tolist(),
                                        layer_name=internal_states.get("layer_name", "Layer"),
                                        head_idx=0
                                    )
                                    logger.info(f"✓ Created attention heatmap from raw data (shape: {attn_2d.shape})")
                                else:
                                    logger.warning(f"Cannot reshape attention weights to square matrix: {len(attn_list)} elements")
                            else:
                                # Already 2D structure
                                attention_fig = viz.create_attention_heatmap(
                                    attention_weights=attn_list,
                                    layer_name=internal_states.get("layer_name", "Layer"),
                                    head_idx=0
                                )
                                logger.info(f"✓ Created attention heatmap from 2D raw data")
                        
                        else:
                            logger.debug(f"Skipping attention data: type={type(attn_data)}, value={attn_data}")
                            
                    except Exception as e:
                        logger.error(f"Failed to process attention data: {e}", exc_info=True)
            
            # Process feature importance (same robust handling)
            if feature_importance_fig is None:
                fi_data = internal_states.get("feature_importance")
                if fi_data is not None:
                    try:
                        if isinstance(fi_data, dict) and "sparse_values" in fi_data:
                            sparse_vals = fi_data.get("sparse_values", [])
                            sparse_idxs = fi_data.get("sparse_indices", [])
                            shape = fi_data.get("shape", [len(sparse_vals)])
                            
                            dense = np.zeros(np.prod(shape), dtype=np.float32)
                            if sparse_idxs and sparse_vals:
                                dense[sparse_idxs] = sparse_vals
                            dense = dense.reshape(shape).flatten()
                            
                            feature_importance_fig = viz.create_feature_importance_chart(
                                feature_importance=dense.tolist(),
                                top_k=10
                            )
                            logger.info(f"✓ Created feature importance from sparse data (shape: {dense.shape})")
                        elif isinstance(fi_data, (list, np.ndarray)) and len(fi_data) > 0:
                            feature_importance_fig = viz.create_feature_importance_chart(
                                feature_importance=fi_data,
                                top_k=10
                            )
                            logger.info("✓ Created feature importance from raw data")
                    except Exception as e:
                        logger.warning(f"Failed to process feature importance: {e}")
            
            # Early exit optimization
            if attention_fig and feature_importance_fig:
                break
        # === END CRITICAL FIX ===
        
        # Build layout
        children = [
            html.H2(f"Analysis: Run {selected_run_id[:8]}"),
            html.Div([
                html.Strong("Model Type:"), f" {summary.get('model_type', 'N/A')} | ",
                html.Strong("Framework:"), f" {summary.get('framework', 'N/A')} | ",
                html.Strong("Mode:"), f" {summary.get('mode', 'N/A')} | ",
                html.Strong("Total Logs:"), f" {summary.get('total_logs', 'N/A')}"
            ], style={"backgroundColor": "#e7f3ff", "padding": "10px", "marginBottom": "20px", "borderRadius": "5px"}),
            
            html.H3("Training Loss Curve", style={"marginTop": "30px"}),
            dcc.Graph(figure=loss_fig),
            
            html.H3("Layer Activation Distribution", style={"marginTop": "30px"}),
            dcc.Graph(figure=activations_fig),
        ]
        
        if attention_fig:
            children.extend([
                html.H3("Attention Heatmap", style={"marginTop": "30px"}),
                dcc.Graph(figure=attention_fig)
            ])
        
        if feature_importance_fig:
            children.extend([
                html.H3("Feature Importance", style={"marginTop": "30px"}),
                dcc.Graph(figure=feature_importance_fig)
            ])
        
        if gradient_fig:
            children.extend([
                html.H3("Gradient Norms by Layer", style={"marginTop": "30px"}),
                dcc.Graph(figure=gradient_fig)
            ])
        
        logger.info("Analysis panel rendering complete")
        return html.Div(children, style={"padding": "20px"})
        
    except Exception as e:
        logger.error(f"Error rendering analysis panel: {e}", exc_info=True)
        return html.Div(f"Error loading analysis: {str(e)}", style={"padding": "20px", "color": "red"})