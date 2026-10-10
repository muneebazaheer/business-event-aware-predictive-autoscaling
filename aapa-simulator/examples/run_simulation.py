#!/usr/bin/env python3
"""
Example script for running autoscaling simulation.

This demonstrates how to:
1. Load a trained classifier
2. Set up the AAPA autoscaler with strategies
3. Run simulations on test workloads
4. Compare against baselines (HPA, Generic Predictive)
5. Calculate and display metrics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import joblib
from typing import Dict, List

from src.core.autoscaler import AAPAAutoscaler
from src.core.strategies import create_strategies
from src.core.simulator import KubernetesSimulator
from src.utils.metrics import calculate_metrics
from src.data.loader import AzureFunctionsDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_simulation_for_workload(
    simulator: KubernetesSimulator,
    workload: np.ndarray,
    autoscaler_name: str
) -> Dict:
    """Run simulation for a single workload and return metrics."""
    logger.info(f"Running simulation with {autoscaler_name}")
    
    # Reset simulator
    simulator.reset()
    
    # Run simulation
    results = simulator.run(workload)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    metrics['autoscaler'] = autoscaler_name
    
    return metrics


def main():
    """Run the simulation pipeline."""
    # Configuration
    model_dir = Path("models")
    data_dir = Path("data/raw")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Load trained classifier
    logger.info("Loading trained classifier...")
    classifier_path = model_dir / "lightgbm_classifier.pkl"
    if not classifier_path.exists():
        logger.error(f"Classifier not found at {classifier_path}")
        logger.error("Please run run_classification.py first")
        return
        
    classifier = joblib.load(classifier_path)
    
    # Create scaling strategies
    logger.info("Creating scaling strategies...")
    strategy_config = {
        'spike': {
            'target_cpu': 0.3,
            'cooldown': 1200,
            'warm_pool_size': 2,
            'min_replicas': 2
        },
        'periodic': {
            'target_cpu': 0.75,
            'cooldown': 180,
            'min_replicas': 1
        },
        'ramp': {
            'target_cpu': 0.6,
            'cooldown': 420,
            'min_replicas': 1
        },
        'stationary_noisy': {
            'target_cpu': 0.55,
            'cooldown': 720,
            'min_replicas': 1
        }
    }
    strategies = create_strategies(strategy_config)
    
    # Create AAPA autoscaler
    logger.info("Creating AAPA autoscaler...")
    aapa = AAPAAutoscaler(classifier, strategies)
    
    # Load test workloads (days 12-14)
    logger.info("Loading test workloads...")
    loader = AzureFunctionsDataLoader(data_dir)
    test_data = loader.load_invocation_data(days=[12, 13, 14])
    
    # Filter and select sample workloads
    test_workloads = []
    for func_hash, time_series in test_data.items():
        if np.sum(time_series) >= 1000:  # Active functions only
            test_workloads.append({
                'function_hash': func_hash,
                'time_series': time_series
            })
    
    # Limit to 100 workloads for demo
    test_workloads = test_workloads[:100]
    logger.info(f"Selected {len(test_workloads)} test workloads")
    
    # Set up simulators
    sim_config = {
        'pod_startup_time': 30,  # seconds
        'cpu_per_request': 0.001,  # CPU units per request
        'max_replicas': 100,
        'metric_interval': 60  # seconds
    }
    
    # Create simulators for each autoscaler
    simulators = {
        'AAPA': KubernetesSimulator(aapa, sim_config),
        'HPA': KubernetesSimulator('hpa', sim_config),
        'Generic_Predictive': KubernetesSimulator('predictive', sim_config)
    }
    
    # Run simulations
    all_results = []
    
    for i, workload_info in enumerate(test_workloads):
        logger.info(f"\\nProcessing workload {i+1}/{len(test_workloads)}")
        
        workload = workload_info['time_series']
        
        for autoscaler_name, simulator in simulators.items():
            metrics = run_simulation_for_workload(
                simulator, workload, autoscaler_name
            )
            metrics['workload_id'] = workload_info['function_hash']
            all_results.append(metrics)
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    # Calculate aggregate metrics by autoscaler
    logger.info("\\nAggregate Results:")
    logger.info("-" * 80)
    
    agg_metrics = results_df.groupby('autoscaler').agg({
        'slo_violation_rate': ['mean', 'std'],
        'mean_response_time': ['mean', 'std'],
        'total_pod_minutes': ['mean', 'std'],
        'mean_cpu_utilization': ['mean', 'std'],
        'scaling_actions': ['mean', 'std']
    }).round(3)
    
    print(agg_metrics)
    
    # Save detailed results
    results_path = results_dir / "simulation_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\\nSaved detailed results to {results_path}")
    
    # Calculate improvements of AAPA over HPA
    logger.info("\\nAAPA Improvements over HPA:")
    logger.info("-" * 80)
    
    aapa_metrics = results_df[results_df['autoscaler'] == 'AAPA'].set_index('workload_id')
    hpa_metrics = results_df[results_df['autoscaler'] == 'HPA'].set_index('workload_id')
    
    common_workloads = aapa_metrics.index.intersection(hpa_metrics.index)
    
    for metric in ['slo_violation_rate', 'mean_response_time', 'total_pod_minutes']:
        aapa_values = aapa_metrics.loc[common_workloads, metric]
        hpa_values = hpa_metrics.loc[common_workloads, metric]
        
        improvement = ((hpa_values - aapa_values) / hpa_values * 100).mean()
        logger.info(f"{metric}: {improvement:.1f}% improvement")
    
    logger.info("\\nSimulation completed successfully!")


if __name__ == "__main__":
    main()