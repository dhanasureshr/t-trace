# test_dashboard_debug.py
import logging
import sys
from pathlib import Path
import os



sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )
)
logging.basicConfig(level=logging.DEBUG)  # ← Enable debug logs

from t_trace.analysis_engine import enable_analysis

enable_analysis(host="127.0.0.1", port=8050, debug=True)