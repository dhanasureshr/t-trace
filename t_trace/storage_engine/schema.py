"""Comprehensive Parquet schema implementation matching M-TRACE specification Section 3.1."""
import pyarrow as pa
import numpy as np
from typing import Dict, Any


def optional(field: pa.Field) -> pa.Field:
    """Mark a field as nullable (optional)."""
    return pa.field(field.name, field.type, nullable=True)


def get_mtrace_schema() -> pa.Schema:
    """
    Return the complete M-TRACE Parquet schema with proper nullability handling.
    
    Required fields (non-nullable):
      - model_metadata
      - internal_states  
      - event_type
    
    Optional fields (nullable) - logged only when enabled in config.yml:
      - contextual_info
      - intermediate_outputs
      - error_analysis
      - uncertainty_sensitivity
      - training_dynamics
      - data_quality
      - modality_specific
      - fusion_mechanisms
      - graph_structure
      - message_passing
      - graph_outputs
      - compression_metadata
      - sparse_logging_metadata
    """
    # ===== MODEL METADATA (REQUIRED) =====
    model_metadata = pa.struct([
        pa.field("model_type", pa.string(), nullable=False),
        pa.field("framework", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("ms"), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("mode", pa.string(), nullable=False),
        pa.field("model_architecture", pa.struct([
            pa.field("num_layers", pa.int32(), nullable=False),
            pa.field("layer_types", pa.list_(pa.string()), nullable=False),
            pa.field("connections", pa.list_(pa.string()), nullable=False),
        ]), nullable=False),
        pa.field("hyperparameters", pa.struct([
            pa.field("learning_rate", pa.float64(), nullable=False),
            pa.field("batch_size", pa.int32(), nullable=False),
            pa.field("optimizer", pa.string(), nullable=False),
            pa.field("other_params", pa.map_(pa.string(), pa.string()), nullable=False),
        ]), nullable=False),
        pa.field("layer_metadata", pa.struct([
            pa.field("layer_type", pa.string(), nullable=False),
            pa.field("activation_function", pa.string(), nullable=False),
            pa.field("num_parameters", pa.int64(), nullable=False),
        ]), nullable=False),
    ])
    
    # ===== INTERNAL STATES (REQUIRED) =====
    internal_states = pa.struct([
        pa.field("layer_name", pa.string(), nullable=False),
        pa.field("layer_index", pa.int32(), nullable=False),  # ← SCALAR int32 (critical!)
        pa.field("attention_weights", pa.list_(pa.float32()), nullable=True),
        pa.field("feature_maps", pa.list_(pa.float32()), nullable=True),
        pa.field("node_splits", pa.list_(pa.string()), nullable=True),
        pa.field("gradients", pa.list_(pa.float32()), nullable=True),
        pa.field("losses", pa.float64(), nullable=False),
        pa.field("feature_importance", pa.list_(pa.float32()), nullable=True),
        pa.field("decision_paths", pa.list_(pa.string()), nullable=True),
    ])
    
    # ===== OPTIONAL SECTIONS (ALL NULLABLE) =====
    contextual_info = optional(pa.field("contextual_info", pa.struct([
        optional(pa.field("input_data", pa.binary())),
        optional(pa.field("output_data", pa.binary())),
    ])))
    
    intermediate_outputs = optional(pa.field("intermediate_outputs", pa.struct([
        optional(pa.field("layer_activations", pa.list_(pa.float32()))),
        optional(pa.field("attention_distributions", pa.list_(pa.float32()))),
    ])))
    
    error_analysis = optional(pa.field("error_analysis", pa.struct([
        optional(pa.field("per_class_metrics", pa.struct([
            optional(pa.field("accuracy", pa.map_(pa.string(), pa.float64()))),
            optional(pa.field("precision", pa.map_(pa.string(), pa.float64()))),
            optional(pa.field("recall", pa.map_(pa.string(), pa.float64()))),
            optional(pa.field("f1_score", pa.map_(pa.string(), pa.float64()))),
        ]))),
        optional(pa.field("confusion_matrix", pa.list_(pa.list_(pa.int64())))),
        optional(pa.field("error_cases", pa.list_(pa.struct([
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("true_label", pa.string(), nullable=False),
            pa.field("predicted_label", pa.string(), nullable=False),
            pa.field("confidence", pa.float64(), nullable=False),
        ])))),
    ])))
    
    uncertainty_sensitivity = optional(pa.field("uncertainty_sensitivity", pa.struct([
        optional(pa.field("uncertainty_estimates", pa.struct([
            optional(pa.field("confidence_scores", pa.list_(pa.float64()))),
            optional(pa.field("prediction_intervals", pa.list_(pa.struct([
                pa.field("lower_bound", pa.float64(), nullable=False),
                pa.field("upper_bound", pa.float64(), nullable=False),
            ])))),
        ]))),
        optional(pa.field("counterfactual_data", pa.list_(pa.struct([
            pa.field("perturbed_input", pa.binary(), nullable=False),
            pa.field("prediction", pa.string(), nullable=False),
            pa.field("difference", pa.float64(), nullable=False),
        ])))),
    ])))
    
    training_dynamics = optional(pa.field("training_dynamics", pa.struct([
        optional(pa.field("learning_rate_schedules", pa.list_(pa.float64()))),
        optional(pa.field("gradient_norms", pa.list_(pa.float64()))),
        optional(pa.field("weight_updates", pa.list_(pa.float64()))),
    ])))
    
    data_quality = optional(pa.field("data_quality", pa.struct([
        optional(pa.field("class_imbalance", pa.map_(pa.string(), pa.float64()))),
        optional(pa.field("missing_values", pa.int64())),
        optional(pa.field("data_drift", pa.struct([
            pa.field("drift_score", pa.float64(), nullable=False),
            pa.field("drift_detected", pa.bool_(), nullable=False),
        ]))),
    ])))
    
    modality_specific = optional(pa.field("modality_specific", pa.struct([
        optional(pa.field("text_attention_weights", pa.list_(pa.float32()))),
        optional(pa.field("text_embeddings", pa.list_(pa.float32()))),
        optional(pa.field("image_feature_maps", pa.list_(pa.float32()))),
        optional(pa.field("image_spatial_attention", pa.list_(pa.float32()))),
        optional(pa.field("audio_spectrogram_features", pa.list_(pa.float32()))),
        optional(pa.field("audio_temporal_attention", pa.list_(pa.float32()))),
    ])))
    
    fusion_mechanisms = optional(pa.field("fusion_mechanisms", pa.struct([
        optional(pa.field("fusion_outputs", pa.list_(pa.float32()))),
        optional(pa.field("modality_contributions", pa.map_(pa.string(), pa.float64()))),
        optional(pa.field("cross_modality_attention", pa.list_(pa.float32()))),
    ])))
    
    graph_structure = optional(pa.field("graph_structure", pa.struct([
        optional(pa.field("node_features", pa.list_(pa.list_(pa.float32())))),
        optional(pa.field("edge_features", pa.list_(pa.list_(pa.float32())))),
        optional(pa.field("graph_features", pa.list_(pa.float32()))),
        optional(pa.field("adjacency_matrix", pa.list_(pa.list_(pa.int32())))),
    ])))
    
    message_passing = optional(pa.field("message_passing", pa.struct([
        optional(pa.field("node_embeddings", pa.list_(pa.list_(pa.float32())))),
        optional(pa.field("edge_embeddings", pa.list_(pa.list_(pa.float32())))),
        optional(pa.field("message_passing_weights", pa.list_(pa.float32()))),
    ])))
    
    graph_outputs = optional(pa.field("graph_outputs", pa.struct([
        optional(pa.field("graph_embeddings", pa.list_(pa.float32()))),
        optional(pa.field("graph_predictions", pa.string())),
        optional(pa.field("node_predictions", pa.list_(pa.string()))),
    ])))
    
    compression_metadata = optional(pa.field("compression_metadata", pa.struct([
        pa.field("algorithm", pa.string(), nullable=False),
        pa.field("level", pa.int8(), nullable=False),
        pa.field("original_size_bytes", pa.int64(), nullable=False),
        pa.field("compressed_size_bytes", pa.int64(), nullable=False),
        pa.field("compression_ratio", pa.float64(), nullable=False),
    ])))
    
    # ===== SPARSE LOGGING METADATA (CRITICAL FIX) =====
    sparse_logging_metadata = optional(pa.field("sparse_logging_metadata", pa.struct([
        pa.field("threshold_applied", pa.float32(), nullable=False),
        pa.field("top_k_values_logged", pa.int32(), nullable=False),      # ← SCALAR count
        pa.field("original_tensor_shape", pa.list_(pa.int32()), nullable=False),
        pa.field("sparse_indices_count", pa.int32(), nullable=False),      # ← SCALAR count
        pa.field("sparse_type", pa.string(), nullable=False),
        pa.field("sparse_indices", pa.list_(pa.int32()), nullable=False),  # ← LIST of indices (FIXED!)
    ])))
    
    # ===== FULL SCHEMA =====
    return pa.schema([
        # Core required fields (non-nullable)
        pa.field("model_metadata", model_metadata, nullable=False),
        pa.field("internal_states", internal_states, nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        
        # Optional sections (nullable)
        contextual_info,
        intermediate_outputs,
        error_analysis,
        uncertainty_sensitivity,
        training_dynamics,
        data_quality,
        modality_specific,
        fusion_mechanisms,
        graph_structure,
        message_passing,
        graph_outputs,
        compression_metadata,
        sparse_logging_metadata,
    ])


def validate_log_entry(log_entry: Dict[str, Any], schema: pa.Schema) -> bool:
    """
    Validate a log entry against the M-TRACE schema with proper null handling.
    
    NOTE: This function is lenient - it fills missing optional fields with None
    before validation to accommodate modular logging configuration.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Create a complete record by filling missing optional fields with None
        complete_record = {}
        for field in schema:
            if field.name in log_entry:
                complete_record[field.name] = log_entry[field.name]
            elif field.nullable:
                complete_record[field.name] = None
            else:
                # Required field missing
                logger.error(f"Required field '{field.name}' missing from log entry")
                return False
        
        # Convert to single-row table for validation
        table = pa.Table.from_pydict(
            {k: [v] for k, v in complete_record.items()},
            schema=schema
        )
        return True
        
    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        logger.debug(f"Invalid log entry keys: {list(log_entry.keys())}")
        return False


def apply_sparse_filtering(tensor_data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply sparse logging filter per Section 3.1.4 (threshold + top-k).
    
    Args:
        tensor_data: Raw tensor/array data (PyTorch, TensorFlow, or NumPy)
        config: Sparse logging configuration dict with keys:
                - sparse_logging.enabled (bool)
                - sparse_logging.sparse_threshold (float)
                - sparse_logging.top_k_values (int)
    
    Returns:
        Dictionary with sparse representation containing:
        - sparse_values: List of significant values
        - sparse_indices: Indices of significant values (FLAT indices)
        - shape: Original tensor shape
        - threshold_applied: Threshold used for filtering
        - sparse_type: "threshold", "top_k", or "all_values"
    """
    # Early exit if sparse logging disabled
    if not config.get("sparse_logging", {}).get("enabled", True):
        if hasattr(tensor_data, "detach"):  # PyTorch tensor
            np_data = tensor_data.detach().cpu().numpy()
        elif hasattr(tensor_data, "numpy"):  # TensorFlow tensor
            np_data = tensor_data.numpy()
        else:
            np_data = np.array(tensor_data)
        
        return {
            "sparse_values": np_data.flatten().tolist(),
            "sparse_indices": np.arange(np_data.size).tolist(),
            "shape": list(np_data.shape),
            "threshold_applied": 0.0,
            "sparse_type": "all_values"
        }
    
    # Convert to numpy if needed
    if hasattr(tensor_data, "detach"):  # PyTorch tensor
        np_data = tensor_data.detach().cpu().numpy()
    elif hasattr(tensor_data, "numpy"):  # TensorFlow tensor
        np_data = tensor_data.numpy()
    else:
        np_data = np.array(tensor_data)
    
    # Handle empty arrays
    if np_data.size == 0:
        return {
            "sparse_values": [],
            "sparse_indices": [],
            "shape": list(np_data.shape),
            "threshold_applied": 0.0,
            "sparse_type": "empty"
        }
    
    abs_values = np.abs(np_data)
    threshold = config.get("sparse_logging", {}).get("sparse_threshold", 0.1)
    top_k = config.get("sparse_logging", {}).get("top_k_values", 5)
    
    # Get indices above threshold
    above_threshold = abs_values > threshold
    indices = np.where(above_threshold)
    
    if len(indices[0]) == 0:
        # Fallback to top-k if nothing above threshold
        flat = abs_values.flatten()
        if len(flat) <= top_k:
            # Return all values if tensor smaller than top_k
            return {
                "sparse_values": np_data.flatten().tolist(),
                "sparse_indices": np.arange(len(flat)).tolist(),
                "shape": list(np_data.shape),
                "threshold_applied": threshold,
                "sparse_type": "all_values"
            }
        
        flat_indices = np.argpartition(flat, -top_k)[-top_k:]
        values = np_data.flatten()[flat_indices]
        return {
            "sparse_values": values.tolist(),
            "sparse_indices": flat_indices.tolist(),  # ← FLAT indices (critical for storage)
            "shape": list(np_data.shape),
            "threshold_applied": threshold,
            "sparse_type": "top_k"
        }
    
    # Return sparse representation from threshold filtering
    values = np_data[indices]
    
    # Convert multi-dimensional indices to FLAT indices for storage efficiency
    flat_indices = np.ravel_multi_index(indices, np_data.shape)
    
    return {
        "sparse_values": values.tolist(),
        "sparse_indices": flat_indices.tolist(),  # ← FLAT indices (critical for storage)
        "shape": list(np_data.shape),
        "threshold_applied": threshold,
        "sparse_type": "threshold"
    }