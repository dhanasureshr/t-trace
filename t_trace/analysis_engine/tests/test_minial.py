"""Test AnalysisEngine dashboard startup and log loading."""
import sys
from pathlib import Path
import os



sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )
)


from t_trace.analysis_engine import enable_analysis

# Start dashboard in development mode
print("🚀 Starting M-TRACE Analysis Dashboard...")
print("   Access at: http://127.0.0.1:8050")
print("   Press Ctrl+C to stop\n")

enable_analysis(
    host="127.0.0.1",
    port=8050,
    debug=True  # Enables hot-reloading during development
)