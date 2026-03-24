
#!/usr/bin/env python3
"""
M-TRACE Artifact Environment Validation
Validates GPU, frameworks, and dependencies for reproducibility
"""
import sys
import json
import platform
from pathlib import Path
from datetime import datetime

def check_python_environment():
    """Verify Python version and critical packages."""
    results = {"python_version": platform.python_version(), "packages": {}}
    
    required = {
        "torch": "2.5.1",
        "tensorflow": "2.20.0", 
        "sklearn": "1.8.0",  # Note: package name is 'sklearn' not 'scikit-learn'
        "pyarrow": "23.0.0",
        "pandas": "3.0.0",
        "numpy": "1.26.4"
    }
    
    # Map import names to package names for version checking
    package_names = {
        "sklearn": "scikit-learn"
    }
    
    for import_name, expected in required.items():
        try:
            module = __import__(import_name)
            actual = getattr(module, "__version__", "unknown")
            
            # For sklearn, get version from scikit-learn
            if import_name == "sklearn":
                import importlib.metadata
                try:
                    actual = importlib.metadata.version("scikit-learn")
                except:
                    actual = getattr(module, "__version__", "unknown")
            
            results["packages"][import_name] = {
                "installed": actual,
                "expected": expected,
                "match": expected.split('+')[0] in actual
            }
        except ImportError as e:
            results["packages"][import_name] = {"installed": None, "expected": expected, "error": str(e)}
    
    return results

def check_gpu_environment():
    """Verify CUDA, cuDNN, and GPU availability."""
    results = {"cuda_available": False, "gpu_info": None}
    
    try:
        import torch
        results["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            results["gpu_info"] = {
                "name": torch.cuda.get_device_name(0),
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version()
            }
    except Exception as e:
        results["gpu_error"] = str(e)
    
    return results

def check_storage_permissions():
    """Verify write access to storage directories."""
    results = {"storage_accessible": True, "errors": []}
    
    test_dirs = [
        Path("mtrace_logs/development"),
        Path("mtrace_logs/production"),
        Path("results/raw")
    ]
    
    for test_dir in test_dirs:
        try:
            test_dir.mkdir(parents=True, exist_ok=True)
            test_file = test_dir / ".write_test"
            test_file.write_text("validation")
            test_file.unlink()
        except Exception as e:
            results["storage_accessible"] = False
            results["errors"].append(f"{test_dir}: {e}")
    
    return results

def main():
    print("🔍 M-TRACE Artifact Environment Validation")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Host: {platform.node()} | OS: {platform.platform()}")
    print(f"Python Executable: {sys.executable}")
    print("=" * 60)
    
    validation = {
        "python": check_python_environment(),
        "gpu": check_gpu_environment(),
        "storage": check_storage_permissions(),
        "overall_pass": True
    }
    
    # Print detailed package status
    print("\n📦 Package Status:")
    for pkg, info in validation["python"]["packages"].items():
        status = "✓" if info.get("match", False) else "✗"
        installed = info.get("installed", "N/A")
        expected = info.get("expected", "N/A")
        print(f"  {status} {pkg:15s} Installed: {str(installed):20s} Expected: {expected}")
        if "error" in info:
            print(f"      Error: {info['error']}")
    
    # Determine overall status
    checks = [
        all(p.get("match", False) for p in validation["python"]["packages"].values()),
        validation["gpu"]["cuda_available"],
        validation["storage"]["storage_accessible"]
    ]
    validation["overall_pass"] = all(checks)
    
    # Output summary
    print(f"\nPython Environment: {'✓ PASS' if all(p.get('match', False) for p in validation['python']['packages'].values()) else '✗ FAIL'}")
    print(f"GPU Environment: {'✓ PASS' if validation['gpu']['cuda_available'] else '✗ FAIL'}")
    print(f"Storage Permissions: {'✓ PASS' if validation['storage']['storage_accessible'] else '✗ FAIL'}")
    
    if validation["gpu"]["gpu_info"]:
        print(f"  GPU: {validation['gpu']['gpu_info']['name']}")
        print(f"  CUDA: {validation['gpu']['gpu_info']['cuda_version']}")
    
    # Save machine-readable report
    report_path = Path("results/validation-report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(validation, f, indent=2, default=str)
    print(f"\n📄 Full report saved to: {report_path.resolve()}")
    
    # Exit with appropriate code
    sys.exit(0 if validation["overall_pass"] else 1)

if __name__ == "__main__":
    main()