#!/usr/bin/env python3
"""
Regenerate worst_parameters.png visualization from the most recent parameter sweep results.
Uses the improved calculation method with normalized scores, convergence rate, and sample size adjustment.
"""
import json
import os
import glob
from datetime import datetime
from evaluation.parameter_sweep import ParameterSweep
from visualization.visualizer import Visualizer

def find_latest_sweep_results():
    """Find the most recent parameter sweep results file."""
    results_files = glob.glob("results/parameter_sweep_results_*.json")
    if not results_files:
        return None
    # Sort by modification time, get most recent
    latest = max(results_files, key=os.path.getmtime)
    return latest

def load_sweep_results(filepath):
    """Load parameter sweep results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    print("="*60)
    print("Regenerating Worst Parameters Visualization")
    print("="*60)
    
    # Find most recent results
    latest_file = find_latest_sweep_results()
    if not latest_file:
        print("❌ No parameter sweep results found in results/ directory")
        print("   Please run parameter sweeps first using: python main.py sweep")
        return
    
    print(f"\n📁 Loading results from: {latest_file}")
    print(f"   Modified: {datetime.fromtimestamp(os.path.getmtime(latest_file)).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load results
    results = load_sweep_results(latest_file)
    print(f"   Found {len(results)} test results")
    
    # Create ParameterSweep instance and load results
    sweep = ParameterSweep(generate_stance_plots=False, parallel=False)
    sweep.results = results
    
    print("\n" + "="*60)
    print("Calculating Worst Parameters (Improved Method)")
    print("="*60)
    print("Using:")
    print("  - Normalized combined score (0.0-1.0)")
    print("  - Error weight: 0.4")
    print("  - Hallucination weight: 0.4")
    print("  - Convergence weight: 0.2")
    print("  - Sample size adjustment")
    print("  - Minimum samples: 3")
    
    # Calculate worst parameters with improved method
    worst_params = sweep.get_worst_parameters(
        error_weight=0.4,
        hallucination_weight=0.4,
        convergence_weight=0.2,
        min_samples=3
    )
    worst_config = sweep.get_worst_config()
    
    if not worst_params:
        print("\n❌ No worst parameters found. Make sure you have sufficient test results.")
        return
    
    # Display worst parameters
    print("\n" + "="*60)
    print("Worst Parameters Found (Most Likely to Cause Hallucinations):")
    print("="*60)
    for param, info in worst_params.items():
        param_name = param.replace("_", " ").title()
        print(f"\n{param_name}: {info['value']}")
        print(f"  Error Rate: {info.get('error_rate', 0):.2%}")
        print(f"  Hallucination Rate: {info.get('hallucination_rate', 0):.2%}")
        print(f"  Non-Convergence Rate: {info.get('non_convergence_rate', 0):.2%}")
        print(f"  Convergence Rate: {info.get('convergence_rate', 0):.2%}")
        print(f"  Correct Rate: {info.get('correct_rate', 0):.2%}")
        print(f"  Sample Size: {info.get('sample_size', 0)} tests")
        if 'combined_score' in info:
            print(f"  Combined Score (normalized): {info['combined_score']:.3f} (0.0 = best, 1.0 = worst)")
    
    # Display worst configuration
    print("\n" + "="*60)
    print("Least Optimized Configuration:")
    print("="*60)
    if worst_config:
        for key, value in worst_config.items():
            if key == "personalities":
                print(f"{key.replace('_', ' ').title()}: {', '.join(value)}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
    else:
        print("(Some parameters may not have been analyzed)")
    
    # Generate visualization
    print("\n" + "="*60)
    print("Generating Visualization")
    print("="*60)
    
    visualizer = Visualizer()
    worst_plot_path = "visualizations/worst_parameters.png"
    
    # Update visualization to include new metrics
    visualizer.plot_worst_parameters(
        worst_params, 
        worst_config,
        save_path=worst_plot_path
    )
    
    print(f"\n✓ Worst parameters visualization saved to: {worst_plot_path}")
    print(f"✓ Visualization includes updated metrics:")
    print(f"  - Normalized combined scores (0.0-1.0)")
    print(f"  - Convergence rates")
    print(f"  - Sample size information")
    
    # Also save updated worst config to file
    config_file = os.path.join("results", f"worst_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    config_data = {
        "worst_config": worst_config,
        "worst_parameters": worst_params,
        "source_file": latest_file,
        "timestamp": datetime.now().isoformat(),
        "calculation_method": "improved_v2",
        "weights": {
            "error_weight": 0.4,
            "hallucination_weight": 0.4,
            "convergence_weight": 0.2
        }
    }
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"\n✓ Worst configuration saved to: {config_file}")

if __name__ == "__main__":
    main()

