#!/usr/bin/env python3
"""
Day 1 Baseline Profiler: Establish performance baselines BEFORE M-TRACE instrumentation
Validated for Adari Workstation (RTX 4080 Super + Ryzen 9 7900X)
"""
import os
import sys
import time
import torch
import torchvision
import pynvml
import numpy as np
from pathlib import Path

# Ensure we're using GPU
assert torch.cuda.is_available(), "CUDA not available!"
device = torch.device("cuda")

# Initialize NVML for GPU monitoring
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_metrics():
    """Capture comprehensive GPU metrics"""
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # %
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    
    return {
        "gpu_util_pct": util,
        "gpu_mem_used_gb": mem.used / 1024**3,
        "gpu_mem_total_gb": mem.total / 1024**3,
        "gpu_power_w": power,
        "gpu_temp_c": temp
    }

def profile_inference(model, input_tensor, iterations=500, warmup=50):
    """Profile inference latency/throughput with GPU synchronization"""
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    
    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Capture final GPU metrics
    gpu_metrics = get_gpu_metrics()
    
    return {
        "latency_ms": (elapsed / iterations) * 1000,
        "throughput_samples_sec": (iterations * input_tensor.size(0)) / elapsed,
        "total_time_sec": elapsed,
        "iterations": iterations,
        **gpu_metrics
    }

def profile_memory(model, input_tensor):
    """Profile peak memory usage during forward pass"""
    torch.cuda.reset_peak_memory_stats()
    _ = model(input_tensor)
    torch.cuda.synchronize()
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
    return {"peak_vram_gb": peak_mem}

def main():
    print("="*70)
    print("M-TRACE PHASE 1 - DAY 1: BASELINE PROFILING")
    print(f"Workstation: Adari Workstation (RTX 4080 Super + Ryzen 9 7900X)")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    print("="*70)
    
    # Create results directory
    results_dir = Path("experiments/phase1/pytorch_baseline/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Test configurations optimized for your 16GB VRAM
    configs = [
        {"model": "resnet50", "batch_size": 64, "input_size": (3, 224, 224)},
        {"model": "resnet50", "batch_size": 128, "input_size": (3, 224, 224)},
        {"model": "bert-base", "batch_size": 32, "input_size": (512,)},  # Sequence length
    ]
    
    results = {}
    
    for cfg in configs:
        print(f"\n▶ Profiling {cfg['model']} (batch={cfg['batch_size']})...")
        
        # Load model
        if cfg["model"] == "resnet50":
            model = torchvision.models.resnet50(pretrained=False).to(device)
            model.eval()
            dummy_input = torch.randn(cfg["batch_size"], *cfg["input_size"]).to(device)
        elif cfg["model"] == "bert-base":
            from transformers import BertModel
            model = BertModel.from_pretrained("bert-base-uncased").to(device)
            model.eval()
            dummy_input = torch.randint(0, 30522, (cfg["batch_size"], cfg["input_size"][0])).to(device)
        
        # Profile inference
        inf_metrics = profile_inference(model, dummy_input)
        
        # Profile memory
        mem_metrics = profile_memory(model, dummy_input)
        
        # Combine results
        results_key = f"{cfg['model']}_bs{cfg['batch_size']}"
        results[results_key] = {
            **inf_metrics,
            **mem_metrics,
            "model": cfg["model"],
            "batch_size": cfg["batch_size"]
        }
        
        print(f"  ✓ Latency: {inf_metrics['latency_ms']:.2f} ms")
        print(f"  ✓ Throughput: {inf_metrics['throughput_samples_sec']:.1f} samples/sec")
        print(f"  ✓ Peak VRAM: {mem_metrics['peak_vram_gb']:.2f} GB")
        print(f"  ✓ GPU Util: {inf_metrics['gpu_util_pct']}% | Temp: {inf_metrics['gpu_temp_c']}°C")
    
    # Save results
    import json
    results_file = results_dir / "baseline_metrics.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✅ Baseline profiling complete!")
    print(f"📊 Results saved to: {results_file}")
    print("\nNext step: Run SAME tests WITH M-TRACE enabled to measure overhead")
    print("="*70)
    
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()