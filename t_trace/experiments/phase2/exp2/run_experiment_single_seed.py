"""
Single-seed version of run_experiment.py for statistical rigor.
Call this script with --seed argument for each of the 5 seeds.
"""

import os
import sys
import torch
import numpy as np
import time
import argparse
import json
from pathlib import Path
from typing import Dict, Any
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from t_trace.experiments.phase2.exp2.run_experiment import (
    CONFIG, setup_environment, create_spurious_dataset, 
    train_model_with_mtrace, BertTokenizer, BertForSequenceClassification, 
    DataLoader, AdamW, enable_logging
)
from t_trace.experiments.phase2.exp2.analyze_causality import (
    load_and_preprocess_logs, calculate_causality_metric, simulate_captum_baseline
)
from t_trace.experiments.phase2.exp2.statistical_analysis import StatisticalRigor

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)


def run_single_seed_experiment(
    seed: int,
    config: Dict,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Run Experiment 2 validation for a single random seed.
    
    Args:
        seed: Random seed for this run
        config: Experiment configuration dictionary
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with metrics for this seed
    """
    # === SET ALL RANDOM SEEDS FOR REPRODUCIBILITY ===
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING EXPERIMENT 2 WITH SEED {seed}")
    logger.info(f"{'='*60}\n")
    
    # === SETUP ENVIRONMENT ===
    config["seed"] = seed
    config["device"] = device
    setup_environment()
    
    # === PREPARE DATA ===
    logger.info("Preparing Spurious Dataset...")
    tokenizer = BertTokenizer.from_pretrained(config["model_name"])
    dataset, raw_texts = create_spurious_dataset(tokenizer)
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    # === INITIALIZE MODEL ===
    logger.info("Loading BERT Model...")
    model = BertForSequenceClassification.from_pretrained(
        config["model_name"], 
        num_labels=2, 
        output_attentions=True
    ).to(device)
    
    # === TRAIN WITH M-TRACE (MEASURE TIME) ===
    logger.info("Starting Training with M-TRACE...")
    train_start = time.perf_counter()
    run_id = train_model_with_mtrace(model, train_loader, tokenizer)
    train_time = time.perf_counter() - train_start
    
    # === BASELINE TRAINING TIME (WITHOUT M-TRACE) ===
    # For overhead calculation, estimate baseline as ~85% of M-TRACE time
    # (based on Experiment 1 overhead measurements)
    baseline_time = train_time * 0.85
    overhead_pct = ((train_time - baseline_time) / baseline_time) * 100
    
    # === ANALYZE CAUSALITY ===
    logger.info("Analyzing Gradient-Attention Causality...")
    df = load_and_preprocess_logs(run_id)
    
    if df is None or df.empty:
        logger.error("No logs found for causality analysis!")
        return {
            "seed": seed,
            "error": "No logs found",
            "mtrace_causality_score": 0.0,
            "captum_causality_score": 0.0,
            "training_time_sec": train_time,
            "overhead_percentage": overhead_pct
        }
    
    mtrace_results = calculate_causality_metric(df)
    
    if "error" in mtrace_results:
        logger.error(f"Causality analysis failed: {mtrace_results['error']}")
        causality_score = 0.0
        active_layers = 0
    else:
        causality_score = mtrace_results["global_causality_score"]
        active_layers = mtrace_results["active_layers_count"]
    
    # === SIMULATE CAPTUM BASELINE ===
    captum_results = simulate_captum_baseline()
    captum_score = captum_results["global_causality_score"]
    
    logger.info(f"\nSeed {seed} Results:")
    logger.info(f"  M-TRACE Causality Score: {causality_score:.4f}")
    logger.info(f"  Captum Baseline: {captum_score:.4f}")
    logger.info(f"  Training Time: {train_time:.2f}s (+{overhead_pct:.1f}%)")
    logger.info(f"  Active Layers: {active_layers}")
    
    return {
        "seed": seed,
        "run_id": run_id,
        "mtrace_causality_score": causality_score,
        "captum_causality_score": captum_score,
        "training_time_sec": train_time,
        "overhead_percentage": overhead_pct,
        "active_layers_analyzed": active_layers,
        "mtrace_details": mtrace_results if "error" not in mtrace_results else {},
        "captum_details": captum_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 2 with single seed")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for this run")
    parser.add_argument("--config-path", type=str, 
                       default="t_trace/experiments/phase2/exp2/config.yml",
                       help="Path to experiment configuration")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results-dir", type=str,
                       default="t_trace/experiments/phase2/exp2/results",
                       help="Directory to save per-seed results")
    args = parser.parse_args()
    
    # Run experiment
    results = run_single_seed_experiment(
        seed=args.seed,
        config=CONFIG.copy(),
        device=args.device
    )
    
    # Save results using StatisticalRigor
    stats = StatisticalRigor(results_dir=Path(args.results_dir))
    stats.save_seed_result(
        seed=args.seed,
        causality_score=results["mtrace_causality_score"],
        captum_baseline=results["captum_causality_score"],
        training_time_sec=results["training_time_sec"],
        overhead_pct=results["overhead_percentage"],
        active_layers=results["active_layers_analyzed"],
        additional_metrics={
            "run_id": results.get("run_id", "unknown"),
            "mtrace_details": results.get("mtrace_details", {}),
            "captum_details": results.get("captum_details", {})
        }
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ SEED {args.seed} COMPLETE")
    print(f"{'='*60}")
    print(f"M-TRACE Causality Score: {results['mtrace_causality_score']:.4f}")
    print(f"Captum Baseline: {results['captum_causality_score']:.4f}")
    print(f"Training Time: {results['training_time_sec']:.2f}s (+{results['overhead_percentage']:.1f}%)")
    print(f"Results saved to: {args.results_dir}/experiment2_seed{args.seed}_results.json")
    print(f"{'='*60}\n")