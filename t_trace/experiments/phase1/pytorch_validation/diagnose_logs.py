#!/usr/bin/env python3
"""Diagnose Parquet schema and overhead root cause"""
import pyarrow.parquet as pq
from pathlib import Path

log_dir = Path("mtrace_logs/development")
latest_parquet = sorted(log_dir.glob("*.parquet"))[-1] if log_dir.exists() else None

if not latest_parquet:
    print("❌ No Parquet files found in mtrace_logs/development/")
    exit(1)

print(f"🔍 Inspecting: {latest_parquet.name}\n")
table = pq.read_table(latest_parquet)

print(f"Rows: {table.num_rows}")
print(f"Columns: {table.column_names}\n")

# Sample first row
print("First log entry (truncated):")
for col in table.column_names[:5]:
    val = table[col][0].as_py() if hasattr(table[col][0], 'as_py') else table[col][0]
    print(f"  {col}: {str(val)[:80]}...")

# Check for layer granularity
if 'internal_states' in table.column_names:
    sample = table['internal_states'][0].as_py()
    print(f"\n⚠️  internal_states structure: {type(sample)}")
    if isinstance(sample, dict):
        print(f"   Keys: {list(sample.keys())[:10]}")
    elif isinstance(sample, list):
        print(f"   List length: {len(sample)}")
        if sample and isinstance(sample[0], dict):
            print(f"   First item keys: {list(sample[0].keys())[:5]}")