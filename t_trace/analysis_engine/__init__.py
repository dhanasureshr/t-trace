"""Public API for M-TRACE AnalysisEngine."""
from .dashboard import AnalysisDashboard

__all__ = ["enable_analysis", "AnalysisDashboard"]

# Global dashboard instance for convenience API
_GLOBAL_DASHBOARD: AnalysisDashboard = None


def enable_analysis(
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    storage_config: dict = None
) -> AnalysisDashboard:
    """
    Public API: Start the M-TRACE Analysis Dashboard.
    
    This function initializes and starts the interactive Dash dashboard for
    visualizing and analyzing M-TRACE logs across all frameworks (PyTorch,
    TensorFlow, scikit-learn).
    
    Args:
        host: Host address to bind the dashboard (default: "127.0.0.1")
        port: Port number to run the dashboard (default: 8050)
        debug: Enable debug mode with hot reloading (default: False)
        storage_config: Optional configuration for StorageEngine integration
        
    Returns:
        Configured AnalysisDashboard instance
    
    Example:
        >>> from mtrace.analysis_engine import enable_analysis
        >>> dashboard = enable_analysis(host="0.0.0.0", port=8050, debug=True)
        >>> # Dashboard accessible at http://0.0.0.0:8050
    """
    global _GLOBAL_DASHBOARD
    
    if _GLOBAL_DASHBOARD is None:
        _GLOBAL_DASHBOARD = AnalysisDashboard(storage_config=storage_config)
    
    _GLOBAL_DASHBOARD.run(host=host, port=port, debug=debug)
    return _GLOBAL_DASHBOARD