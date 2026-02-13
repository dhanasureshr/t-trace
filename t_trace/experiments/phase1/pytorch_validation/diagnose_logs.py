#!/usr/bin/env python3
"""Verify saved Parquet file contains layer_name and valid schema"""
import pyarrow.parquet as pq
from pathlib import Path

# Find latest Parquet file
log_dir = Path("mtrace_logs/development")
latest = sorted(log_dir.glob("*.parquet"), key=lambda f: f.stat().st_mtime)[-1]
print(f"Analyzing: {latest.name}")

# Read table
table = pq.read_table(latest)
print(f"✓ Rows: {table.num_rows}")
print(f"✓ Columns: {table.column_names[:5]}...")

# Sample first log
first_row = table.slice(0, 1).to_pydict()
layer_name = first_row["internal_states"][0].get("layer_name", "MISSING")
print(f"✓ First layer name: {layer_name}")
print(f"✓ Schema validation: {'layer_name' in first_row['internal_states'][0]}")