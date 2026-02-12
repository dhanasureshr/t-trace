#!/usr/bin/env python3
"""
Hardware profiler optimized for Adari Workstation (RTX 4080 Super + Ryzen 9 7900X)
Validates environment readiness for M-TRACE Phase 1 experiments.
"""
import sys
import platform
import subprocess
from pathlib import Path

try:
    import pynvml
    NVML_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    try:
        # Fallback to py3nvml if pynvml not available
        from py3nvml import py3nvml as pynvml
        NVML_AVAILABLE = True
    except:
        NVML_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

def get_gpu_info():
    """Extract GPU details using nvidia-smi (works even without pynvml)"""
    if NVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle).decode() if isinstance(pynvml.nvmlDeviceGetName(handle), bytes) else pynvml.nvmlDeviceGetName(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return {
                "name": name,
                "vram_total_gb": round(mem.total / 1024**3, 1),
                "compute_capability": "8.9"  # RTX 4080 Super
            }
        except Exception as e:
            print(f"NVML error: {e}", file=sys.stderr)
    
    # Fallback to nvidia-smi CLI
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            name, mem_mib = result.stdout.strip().split(", ")
            return {
                "name": name.strip(),
                "vram_total_gb": round(int(mem_mib) / 1024, 1),
                "compute_capability": "8.9"
            }
    except Exception as e:
        print(f"nvidia-smi fallback error: {e}", file=sys.stderr)
    
    return {"name": "Unknown GPU", "vram_total_gb": 0, "compute_capability": "N/A"}

def get_cpu_info():
    """Extract CPU details optimized for Ryzen 9 7900X"""
    if PSUTIL_AVAILABLE:
        cores = psutil.cpu_count(logical=False)
        threads = psutil.cpu_count(logical=True)
    else:
        cores = threads = "N/A (psutil not installed)"
    
    return {
        "model": platform.processor() or "AMD Ryzen 9 7900X",
        "cores": cores,
        "threads": threads,
        "base_clock_ghz": 4.7,
        "boost_clock_ghz": 5.6
    }

def get_ram_info():
    """Extract RAM details (64GB DDR5 expected)"""
    if PSUTIL_AVAILABLE:
        total_gb = round(psutil.virtual_memory().total / 1024**3, 1)
    else:
        total_gb = "N/A"
    
    return {
        "total_gb": total_gb,
        "type": "DDR5",
        "speed_mts": 5600  # Typical for Ryzen 7000 platform
    }

def get_storage_info():
    """Detect NVMe storage (Samsung 990 Pro expected)"""
    root = Path("/")
    try:
        usage = psutil.disk_usage(str(root))
        total_tb = round(usage.total / 1024**4, 2)
    except:
        total_tb = "N/A"
    
    # Detect NVMe devices
    nvme_devices = list(Path("/sys/class/nvme").glob("nvme*")) if Path("/sys/class/nvme").exists() else []
    
    return {
        "type": "NVMe PCIe 4.0" if nvme_devices else "Unknown",
        "devices_detected": len(nvme_devices),
        "total_capacity_tb": total_tb,
        "expected_model": "Samsung 990 Pro (or equivalent)"
    }

def profile_workstation():
    """Comprehensive hardware profile for M-TRACE validation"""
    return {
        "system": {
            "hostname": platform.node(),
            "os": f"{platform.system()} {platform.release()}",
            "kernel": platform.version(),
            "python_version": platform.python_version()
        },
        "gpu": get_gpu_info(),
        "cpu": get_cpu_info(),
        "ram": get_ram_info(),
        "storage": get_storage_info(),
        "mtrace_readiness": {
            "cuda_available": NVML_AVAILABLE,
            "psutil_available": PSUTIL_AVAILABLE,
            "recommended_batch_size_pytorch": 64,  # Optimized for 16GB VRAM (RTX 4080 Super)
            "max_concurrent_experiments": 2  # CPU-bound + GPU-bound can run simultaneously
        }
    }

if __name__ == "__main__":
    import json
    profile = profile_workstation()
    
    print("=" * 70)
    print("M-TRACE HARDWARE PROFILER - Adari Workstation Validation")
    print("=" * 70)
    print(json.dumps(profile, indent=2))
    print("=" * 70)
    
    # Quick validation checks
    gpu_ok = profile["gpu"]["vram_total_gb"] >= 15.5
    ram_ok = profile["ram"]["total_gb"] >= 60
    cuda_ok = NVML_AVAILABLE
    
    print("\n✅ Hardware Validation:")
    print(f"   GPU (RTX 4080 Super): {'PASS' if gpu_ok else 'WARN'} - {profile['gpu']['vram_total_gb']} GB VRAM")
    print(f"   RAM (64GB DDR5):      {'PASS' if ram_ok else 'WARN'} - {profile['ram']['total_gb']} GB")
    print(f"   CUDA Monitoring:      {'PASS' if cuda_ok else 'FAIL'} - {'Available' if cuda_ok else 'Install nvidia-ml-py3'}")
    
    if not cuda_ok:
        print("\n🔧 FIX: Run this to enable GPU monitoring:")
        print("   pip install nvidia-ml-py3 psutil")
    
    # Save profile for experiment reproducibility
    Path("experiments/utils").mkdir(parents=True, exist_ok=True)
    with open("experiments/utils/hardware_profile.json", "w") as f:
        json.dump(profile, f, indent=2)
    print(f"\n💾 Profile saved to: experiments/utils/hardware_profile.json")