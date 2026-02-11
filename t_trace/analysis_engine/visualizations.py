"""Visualization components for M-TRACE AnalysisEngine with explicit empty array safety."""
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def _is_empty_array(arr: Any) -> bool:
    """Safely check if array-like object is empty (handles None, list, tuple, np.ndarray)."""
    if arr is None:
        return True
    if isinstance(arr, (list, tuple)):
        return len(arr) == 0
    if isinstance(arr, np.ndarray):
        return arr.size == 0
    # For scalar values or other types, consider non-empty
    return False


def _safe_array_to_list(arr: Any) -> List:
    """Safely convert array-like to list, handling empties and scalars."""
    if _is_empty_array(arr):
        return []
    if isinstance(arr, (list, tuple)):
        return list(arr)
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    # Handle scalar values
    return [float(arr)] if np.isscalar(arr) else []


def _safe_array_mean(arr: Any) -> float:
    """Safely compute mean of array-like object, handling empties."""
    try:
        values = _safe_array_to_list(arr)
        if not values:
            return 0.0
        return float(np.mean(values))
    except Exception:
        return 0.0


def _safe_array_std(arr: Any) -> float:
    """Safely compute std of array-like object, handling empties."""
    try:
        values = _safe_array_to_list(arr)
        if not values:
            return 0.0
        return float(np.std(values))
    except Exception:
        return 0.0


def _extract_field_safe(row: pd.Series, field_path: List[str], default=None):
    """Safely extract nested field from log row."""
    try:
        value = row
        for field in field_path:
            if isinstance(value, dict):
                value = value.get(field, default)
            elif hasattr(value, 'get'):
                value = value.get(field, default)
            else:
                return default
        return value
    except Exception as e:
        logger.debug(f"Field extraction failed for {field_path}: {e}")
        return default


class Visualizations:
    """Collection of visualization methods for M-TRACE logs with robust error handling."""
    
    @staticmethod
    def create_attention_heatmap(
        attention_weights: List[float],
        tokens: Optional[List[str]] = None,
        layer_name: str = "Layer",
        head_idx: int = 0
    ) -> go.Figure:
        """
        Create attention heatmap visualization for transformer models.
        Handles empty/invalid inputs gracefully.
        """
        try:
            # Handle empty inputs safely
            if _is_empty_array(attention_weights):
                logger.warning("Empty attention weights - returning placeholder heatmap")
                weights = np.zeros((5, 5))
            else:
                weights = np.array(attention_weights)
            
            # Reshape if flattened (assume square matrix)
            if weights.ndim == 1:
                seq_len = int(np.sqrt(len(weights)))
                if seq_len * seq_len != len(weights):
                    logger.warning(f"Cannot reshape attention weights to square matrix: {len(weights)} elements")
                    seq_len = int(np.sqrt(len(weights)))
                    weights = weights[:seq_len * seq_len]
                weights = weights.reshape(seq_len, seq_len)
            
            # Generate tokens if not provided
            if tokens is None:
                tokens = [f"Token {i}" for i in range(weights.shape[1])]
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=weights,
                x=tokens[:weights.shape[1]],
                y=tokens[:weights.shape[0]],
                colorscale="RdBu",
                colorbar=dict(title="Attention Weight"),
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=f"Attention Heatmap - {layer_name} (Head {head_idx})",
                xaxis_title="Target Tokens",
                yaxis_title="Source Tokens",
                height=500,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating attention heatmap: {e}", exc_info=True)
            fig = go.Figure()
            fig.add_annotation(
                text=f"⚠️ Error generating heatmap:<br>{str(e)[:100]}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red"),
                align="center"
            )
            fig.update_layout(
                title="Attention Heatmap (Error)",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            return fig
    
    @staticmethod
    def create_layer_activations_chart(
        logs_df: pd.DataFrame,
        layer_names: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create line chart showing activation distributions across layers.
        Handles empty arrays and missing fields gracefully.
        """
        try:
            if logs_df is None or logs_df.empty:
                logger.warning("Empty DataFrame passed to create_layer_activations_chart")
                return Visualizations._empty_figure("No data available for layer activations")
            
            mean_activations = []
            std_activations = []
            valid_layer_names = []
            
            for idx, row in logs_df.iterrows():
                # Extract output safely
                output = _extract_field_safe(row, ["internal_states", "output"], {})
                
                # Handle sparse representation
                if isinstance(output, dict) and "sparse_values" in output:
                    values = output["sparse_values"]
                elif isinstance(output, (list, np.ndarray, tuple)):
                    values = output
                else:
                    values = []
                
                # Skip empty arrays using EXPLICIT check
                if _is_empty_array(values):
                    continue
                
                # Compute stats safely
                mean_val = _safe_array_mean(values)
                std_val = _safe_array_std(values)
                
                mean_activations.append(mean_val)
                std_activations.append(std_val)
                valid_layer_names.append(_extract_field_safe(row, ["internal_states", "layer_name"], f"Layer {idx}"))
            
            if not mean_activations:
                logger.warning("No valid activations found in logs")
                return Visualizations._empty_figure("No activation data available")
            
            # Create chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=valid_layer_names,
                y=mean_activations,
                mode='lines+markers',
                name='Mean Activation',
                line=dict(color='blue', width=2),
                error_y=dict(
                    type='data',
                    array=std_activations,
                    visible=True,
                    color='lightblue'
                )
            ))
            
            fig.update_layout(
                title="Layer Activation Distribution",
                xaxis_title="Layer",
                yaxis_title="Activation Value",
                height=400,
                hovermode='x unified',
                margin=dict(l=50, r=20, t=40, b=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating layer activations chart: {e}", exc_info=True)
            return Visualizations._error_figure("Layer Activations", str(e))
    
    @staticmethod
    def create_feature_importance_chart(
        feature_importance: List[float],
        feature_names: Optional[List[str]] = None,
        top_k: int = 10
    ) -> go.Figure:
        """
        Create bar chart for feature importance (tree-based models).
        Handles empty arrays safely.
        """
        try:
            if _is_empty_array(feature_importance):
                logger.warning("Empty feature importance array")
                return Visualizations._empty_figure("No feature importance data available")
            
            # Convert to numpy for safe indexing
            importance_array = np.array(feature_importance)
            
            # Handle case where array smaller than top_k
            k = min(top_k, len(importance_array))
            if k == 0:
                return Visualizations._empty_figure("No feature importance data available")
            
            top_indices = np.argsort(importance_array)[-k:][::-1]
            top_importance = importance_array[top_indices]
            
            # Generate feature names if not provided
            if feature_names:
                top_features = [feature_names[i] if i < len(feature_names) else f"Feature {i}" 
                               for i in top_indices]
            else:
                top_features = [f"Feature {i}" for i in top_indices]
            
            # Create horizontal bar chart
            fig = go.Figure(go.Bar(
                x=top_importance,
                y=top_features,
                orientation='h',
                marker=dict(color='steelblue')
            ))
            
            fig.update_layout(
                title=f"Top {k} Feature Importance",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=400,
                margin=dict(l=150, r=50, t=50, b=50),
                yaxis=dict(autorange="reversed")
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature importance chart: {e}", exc_info=True)
            return Visualizations._error_figure("Feature Importance", str(e))
    
    @staticmethod
    def create_loss_curve(
        logs_df: pd.DataFrame,
        window_size: int = 10
    ) -> go.Figure:
        """
        Create loss curve visualization with moving average.
        Handles missing/empty loss values with EXPLICIT checks.
        """
        try:
            if logs_df is None or logs_df.empty:
                logger.warning("Empty DataFrame passed to create_loss_curve")
                return Visualizations._empty_figure("No loss data available")
            
            losses = []
            timestamps = []
            
            for idx, row in logs_df.iterrows():
                # Extract loss safely from nested structure
                loss = _extract_field_safe(row, ["internal_states", "losses"])
                
                # Skip None or empty arrays using EXPLICIT check
                if loss is None or _is_empty_array(loss):
                    continue
                
                # Convert to float safely
                try:
                    loss_val = float(loss) if np.isscalar(loss) else float(loss[0])
                    losses.append(loss_val)
                    timestamps.append(float(_extract_field_safe(row, ["model_metadata", "timestamp"], idx)))
                except (TypeError, ValueError) as e:
                    logger.debug(f"Skipping invalid loss value: {loss} ({e})")
                    continue
            
            if not losses:
                logger.warning("No valid loss values found in logs")
                return Visualizations._empty_figure("No loss values available")
            
            fig = go.Figure()
            
            # Raw losses
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=losses,
                mode='markers',
                name='Raw Loss',
                marker=dict(size=4, color='lightgray'),
                opacity=0.5
            ))
            
            # Moving average (only if enough data points)
            if len(losses) >= window_size and window_size > 1:
                try:
                    moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                    fig.add_trace(go.Scatter(
                        x=timestamps[window_size-1:],
                        y=moving_avg,
                        mode='lines',
                        name=f'Moving Avg ({window_size})',
                        line=dict(color='red', width=2)
                    ))
                except Exception as e:
                    logger.warning(f"Moving average calculation failed: {e}")
            
            fig.update_layout(
                title="Training Loss Curve",
                xaxis_title="Timestamp",
                yaxis_title="Loss",
                height=400,
                hovermode='x unified',
                margin=dict(l=50, r=20, t=40, b=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating loss curve: {e}", exc_info=True)
            return Visualizations._error_figure("Loss Curve", str(e))
    
    @staticmethod
    def create_gradient_norm_chart(
        logs_df: pd.DataFrame
    ) -> go.Figure:
        """
        Create gradient norm visualization for development mode logs.
        Handles missing gradient data with EXPLICIT empty checks.
        """
        try:
            if logs_df is None or logs_df.empty:
                logger.warning("Empty DataFrame passed to create_gradient_norm_chart")
                return Visualizations._empty_figure("No gradient data available")
            
            gradient_norms = []
            layer_names = []
            
            for idx, row in logs_df.iterrows():
                # Only process backward pass logs
                event_type = _extract_field_safe(row, ["event_type"])
                if event_type != "backward":
                    continue
                
                # Extract gradients safely
                gradients = _extract_field_safe(row, ["internal_states", "gradients"], [])
                
                # Handle sparse representation
                if isinstance(gradients, dict) and "sparse_values" in gradients:
                    values = gradients["sparse_values"]
                elif isinstance(gradients, (list, np.ndarray, tuple)):
                    values = gradients
                else:
                    values = []
                
                # Skip empty gradients using EXPLICIT check
                if _is_empty_array(values):
                    continue
                
                # Compute norm safely
                try:
                    norm = np.linalg.norm(np.array(values))
                    gradient_norms.append(float(norm))
                    layer_names.append(_extract_field_safe(row, ["internal_states", "layer_name"], f"Layer {idx}"))
                except Exception as e:
                    logger.debug(f"Skipping gradient norm calculation: {e}")
                    continue
            
            if not gradient_norms:
                logger.warning("No gradient norms found in logs (development mode required)")
                return Visualizations._empty_figure(
                    "No gradient data available<br>(requires development mode logging)"
                )
            
            fig = go.Figure(go.Bar(
                x=layer_names,
                y=gradient_norms,
                marker=dict(color='orange')
            ))
            
            fig.update_layout(
                title="Gradient Norms by Layer (Development Mode)",
                xaxis_title="Layer",
                yaxis_title="Gradient Norm",
                height=400,
                margin=dict(b=100, l=50, r=20, t=40),
                xaxis=dict(tickangle=45)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating gradient norm chart: {e}", exc_info=True)
            return Visualizations._error_figure("Gradient Norms", str(e))
    
    @staticmethod
    def _empty_figure(message: str) -> go.Figure:
        """Create placeholder figure for empty data."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666"),
            align="center"
        )
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='white'
        )
        return fig
    
    @staticmethod
    def _error_figure(title: str, error: str) -> go.Figure:
        """Create error figure with diagnostic message."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ Error in {title}<br><small>{error[:100]}...</small>",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red"),
            align="center"
        )
        fig.update_layout(
            title=f"{title} (Error)",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig