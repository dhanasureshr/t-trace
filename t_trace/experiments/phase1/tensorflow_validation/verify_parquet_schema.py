#!/usr/bin/env python3
"""Diagnose exact Parquet schema structure to confirm nested fields"""
import pyarrow.parquet as pq
from pathlib import Path
import time

log_dir = Path("mtrace_logs/development")
recent_files = [
    f for f in log_dir.glob("*.parquet") 
    if time.time() - f.stat().st_mtime < 120
]
if not recent_files:
    print("❌ No recent Parquet files found")
    exit(1)

latest = max(recent_files, key=lambda f: f.stat().st_mtime)
print(f"🔍 Analyzing: {latest.name}\n")

table = pq.read_table(latest)
print(f"✓ Total rows: {table.num_rows}")
print(f"✓ Top-level columns: {table.column_names}\n")

# Check if internal_states is a struct type
if "internal_states" in table.column_names:
    col = table.column("internal_states")
    print(f"✓ internal_states type: {col.type}")
    
    # Check nested fields
    if hasattr(col.type, 'num_fields'):
        print(f"✓ Nested fields in internal_states ({col.type.num_fields}):")
        for i in range(col.type.num_fields):
            field = col.type.field(i)
            print(f"  • {field.name}: {field.type} (nullable={field.nullable})")
        
        # Verify required fields exist
        required = ["layer_name", "layer_index", "losses"]
        for req in required:
            if req in [col.type.field(i).name for i in range(col.type.num_fields)]:
                print(f"  ✅ Required field '{req}' present")
            else:
                print(f"  ❌ MISSING required field '{req}'")
    else:
        print("⚠️ internal_states is not a struct type (schema violation!)")
else:
    print("❌ internal_states column missing (critical schema violation)")

print("\n✅ Schema diagnosis complete")