"""
Main entry point for the multi-agent conversation system.
"""
import os
import sys
from conversation.conversation_manager import ConversationManager
from judge.judge_agent import JudgeAgent
from data.dataset import Claim, ClaimType, get_claims_by_type, get_all_claims
from evaluation.parameter_sweep import ParameterSweep
from evaluation.advanced_metrics import calculate_conversation_quality_metrics
from visualization.visualizer import Visualizer
from results.debate_storage import save_debate_summary
import config
import json
from datetime import datetime

def run_single_conversation(claim: Claim, personalities: list = None):
    """Run a single conversation and evaluate it."""
    print("\n" + "="*60)
    print("Running Single Conversation")
    print("="*60)
    
    manager = ConversationManager(personalities=personalities)
    result = manager.run_conversation(claim)
    
    print("\n" + "="*60)
    print("Judge Evaluation")
    print("="*60)
    
    judge = JudgeAgent()
    evaluation = judge.evaluate_conversation(
        claim=result["claim"],
        claim_type=result["claim_type"],
        final_responses=result["final_responses"],
        conversation_history=result["conversation_history"]
    )
    
    # Calculate advanced metrics
    advanced_metrics = calculate_conversation_quality_metrics(result, evaluation)
    
    print(f"\nOverall Correctness: {evaluation.get('overall_correctness', 'unknown')}")
    print(f"Converged: {evaluation.get('converged', False)}")
    print(f"Hallucinations Detected: {evaluation.get('hallucinations_detected', False)}")
    print(f"\n--- Advanced Metrics ---")
    print(f"Final Agreement Score: {advanced_metrics.get('final_agreement_score', 0.0):.2f}")
    print(f"Average Agreement: {advanced_metrics.get('average_agreement', 0.0):.2f}")
    print(f"Stance Flips: {advanced_metrics.get('num_stance_flips', 0)}")
    print(f"Average Confidence: {advanced_metrics.get('average_confidence', 0.0):.2f}")
    print(f"Persuasion Events: {len(advanced_metrics.get('persuasion_events', []))}")
    print(f"\nReasoning:\n{evaluation.get('raw_evaluation', 'N/A')}")
    
    # Save result
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"logs/conversation_{timestamp}.json"
    
    full_result = {
        "conversation": result,
        "evaluation": evaluation,
        "advanced_metrics": advanced_metrics
    }
    
    with open(result_file, 'w') as f:
        json.dump(full_result, f, indent=2)
    
    print(f"\nResults saved to {result_file}")
    
    summary_path = save_debate_summary(
        result,
        evaluation,
        timestamp=timestamp,
        metadata={"mode": "single_conversation"}
    )
    print(f"Debate summary saved to {summary_path}")
    
    # Visualize stance shifts (static and interactive)
    visualizer = Visualizer()
    
    # Static visualization
    vis_path = "visualizations/stance_shifts_single_conversation.png"
    visualizer.plot_stance_shifts(result, save_path=vis_path)
    
    # Interactive visualizations
    print("\nGenerating interactive visualizations...")
    visualizer.plot_interactive_stance_shifts(
        result, 
        save_path="visualizations/interactive_stance_shifts.html"
    )
    visualizer.plot_interactive_agreement_tracking(
        result,
        save_path="visualizations/interactive_agreement.html"
    )
    visualizer.plot_interactive_influence_matrix(
        result,
        save_path="visualizations/interactive_influence.html"
    )
    visualizer.plot_interactive_metrics_dashboard(
        result,
        evaluation,
        advanced_metrics,
        save_path="visualizations/interactive_dashboard.html"
    )
    print("Interactive visualizations saved to visualizations/ directory")
    
    return full_result

def run_parameter_sweeps():
    """Run parameter sweeps as specified in the project proposal."""
    print("\n" + "="*60)
    print("Running Parameter Sweeps")
    print("="*60)
    
    sweep = ParameterSweep()
    visualizer = Visualizer()
    
    # Get a sample claim from each category
    sample_claims = {
        ClaimType.GROUND_TRUTH: get_claims_by_type(ClaimType.GROUND_TRUTH)[0],
        ClaimType.FALSE: get_claims_by_type(ClaimType.FALSE)[0],
        ClaimType.DEBATABLE: get_claims_by_type(ClaimType.DEBATABLE)[0]
    }
    
    # 1. Sweep context size
    print("\n--- Sweeping Context Size ---")
    context_sizes = [1000, 2000, 4000]
    context_results_all = []
    # Use consistent personality combination with deceiver for all parameter sweeps
    personalities = ["skeptic", "optimist", "persuader", "deceiver"]
    for claim_type, claim in sample_claims.items():
        results = sweep.sweep_context_size(
            context_sizes=context_sizes,
            claim=claim,
            personalities=personalities
        )
        context_results_all.extend(results)
        visualizer.plot_accuracy_vs_parameter(
            results, 
            "context_size",
            save_path=f"visualizations/accuracy_vs_context_size_{claim_type.value}.png"
        )
    visualizer.plot_average_stance_shifts(
        context_results_all,
        "context_size",
        save_dir="visualizations"
    )
    
    # 2. Sweep max turns
    print("\n--- Sweeping Max Turns ---")
    max_turns_list = [5, 10, 15]
    max_turn_results_all = []
    # Use consistent personality combination with deceiver for all parameter sweeps
    personalities = ["skeptic", "optimist", "persuader", "deceiver"]
    for claim_type, claim in sample_claims.items():
        results = sweep.sweep_max_turns(
            max_turns_list=max_turns_list,
            claim=claim,
            personalities=personalities
        )
        max_turn_results_all.extend(results)
        visualizer.plot_accuracy_vs_parameter(
            results,
            "max_turns",
            save_path=f"visualizations/accuracy_vs_max_turns_{claim_type.value}.png"
        )
    visualizer.plot_average_stance_shifts(
        max_turn_results_all,
        "max_turns",
        save_dir="visualizations"
    )
    
    # 3. Sweep personality combinations
    print("\n--- Sweeping Personality Combinations ---")
    personality_combinations = [
        # Base combinations without deceiver
        ["skeptic", "optimist"],
        ["skeptic", "persuader"],
        ["optimist", "persuader"],
        ["skeptic", "optimist", "persuader"],
        # Deceiver combinations - testing misinformation spread
        ["deceiver", "optimist"],  # Deceiver with trusting optimist
        ["deceiver", "skeptic"],  # Deceiver with skeptical agent
        ["deceiver", "persuader"],  # Deceiver with persuasive agent
        ["skeptic", "deceiver"],  # Alternative order
        ["optimist", "deceiver"],  # Alternative order
        ["persuader", "deceiver"],  # Alternative order
        # Three-agent combinations with deceiver
        ["skeptic", "optimist", "deceiver"],  # Deceiver with skeptic and optimist
        ["skeptic", "persuader", "deceiver"],  # Deceiver with skeptic and persuader
        ["optimist", "persuader", "deceiver"],  # Deceiver with optimist and persuader
        # All four agents
        ["skeptic", "optimist", "persuader", "deceiver"]
    ]
    claim = sample_claims[ClaimType.DEBATABLE]  # Use debatable for personality test
    results = sweep.sweep_personality_combinations(
        personality_combinations=personality_combinations,
        claim=claim
    )
    visualizer.plot_accuracy_vs_parameter(
        results,
        "personality_combinations",
        save_path="visualizations/accuracy_vs_personalities.png"
    )
    visualizer.plot_average_stance_shifts(
        results,
        "personality_combinations",
        save_dir="visualizations"
    )
    
    # 4. Sweep claim types
    print("\n--- Sweeping Claim Types ---")
    all_claims = get_all_claims()
    # Use consistent personality combination with deceiver for all parameter sweeps
    results = sweep.sweep_claim_types(
        claims=all_claims[:5],  # Use first 5 claims for speed
        personalities=["skeptic", "optimist", "persuader", "deceiver"]
    )
    visualizer.plot_average_stance_shifts(
        results,
        "claim_type",
        save_dir="visualizations"
    )
    
    # Generate summary visualizations (static)
    visualizer.plot_convergence_analysis(
        sweep.results,
        save_path="visualizations/convergence_analysis.png"
    )
    visualizer.plot_hallucination_detection(
        sweep.results,
        save_path="visualizations/hallucination_detection.png"
    )
    
    # Generate interactive visualizations for parameter sweep results
    if sweep.results:
        print("\nGenerating interactive visualizations for parameter sweep results...")
        
        # Find interesting results: one with highest hallucination, one with lowest agreement
        results_with_metrics = []
        for result in sweep.results:
            conv = result.get("conversation_result", {})
            eval_result = result.get("evaluation", {})
            if conv and eval_result:
                metrics = calculate_conversation_quality_metrics(conv, eval_result)
                results_with_metrics.append((result, metrics))
        
        if results_with_metrics:
            # Find result with highest hallucination rate (most interesting for analysis)
            max_hallucination_result = max(
                results_with_metrics,
                key=lambda x: 1.0 if x[0]["evaluation"].get("hallucinations_detected", False) else 0.0
            )
            
            # Find result with lowest agreement (most disagreement)
            min_agreement_result = min(
                results_with_metrics,
                key=lambda x: x[1].get("final_agreement_score", 1.0)
            )
            
            # Generate dashboard for highest hallucination result
            result, metrics = max_hallucination_result
            conv = result["conversation_result"]
            eval_result = result["evaluation"]
            visualizer.plot_interactive_metrics_dashboard(
                conv,
                eval_result,
                metrics,
                save_path="visualizations/interactive_sweep_dashboard_hallucination.html"
            )
            print("  - Interactive dashboard (highest hallucination) saved to interactive_sweep_dashboard_hallucination.html")
            
            # Generate dashboard for lowest agreement result
            result, metrics = min_agreement_result
            conv = result["conversation_result"]
            eval_result = result["evaluation"]
            visualizer.plot_interactive_metrics_dashboard(
                conv,
                eval_result,
                metrics,
                save_path="visualizations/interactive_sweep_dashboard_disagreement.html"
            )
            print("  - Interactive dashboard (lowest agreement) saved to interactive_sweep_dashboard_disagreement.html")
            
            # Also generate individual visualizations for the most interesting result
            result, metrics = max_hallucination_result
            conv = result["conversation_result"]
            visualizer.plot_interactive_stance_shifts(
                conv,
                save_path="visualizations/interactive_sweep_stance_shifts.html"
            )
            visualizer.plot_interactive_agreement_tracking(
                conv,
                save_path="visualizations/interactive_sweep_agreement.html"
            )
            visualizer.plot_interactive_influence_matrix(
                conv,
                save_path="visualizations/interactive_sweep_influence.html"
            )
            print("  - Individual interactive plots saved (stance shifts, agreement, influence)")
        
        print("Interactive visualizations for parameter sweeps completed!")
    
    # Save all results
    sweep.save_results()
    
    # 5. Sweep temperature
    print("\n--- Sweeping Temperature ---")
    temperatures = [0.3, 0.5, 0.7, 0.9]
    temp_results_all = []
    # Use consistent personality combination with deceiver for all parameter sweeps
    personalities = ["skeptic", "optimist", "persuader", "deceiver"]
    for claim_type, claim in sample_claims.items():
        results = sweep.sweep_temperature(
            temperatures=temperatures,
            claim=claim,
            personalities=personalities
        )
        temp_results_all.extend(results)
        visualizer.plot_accuracy_vs_parameter(
            results,
            "temperature",
            save_path=f"visualizations/accuracy_vs_temperature_{claim_type.value}.png"
        )
    visualizer.plot_average_stance_shifts(
        temp_results_all,
        "temperature",
        save_dir="visualizations"
    )
    
    # Get worst parameters (most likely to cause hallucinations)
    worst_params = sweep.get_worst_parameters()
    worst_config = sweep.get_worst_config()
    
    # Display worst parameters
    print("\n" + "="*60)
    print("Worst Parameters Found (Most Likely to Cause Hallucinations):")
    print("="*60)
    for param, info in worst_params.items():
        param_name = param.replace("_", " ").title()
        print(f"{param_name}: {info['value']}")
        print(f"  Error Rate: {info.get('error_rate', 0):.2%}")
        print(f"  Hallucination Rate: {info.get('hallucination_rate', 0):.2%}")
        print(f"  Correct Rate: {info.get('correct_rate', 0):.2%}")
        if 'combined_score' in info:
            print(f"  Combined Score: {info['combined_score']:.2f}")
    
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
    
    # Save worst configuration to file
    if worst_config:
        config_file = os.path.join("results", f"worst_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        config_data = {
            "worst_config": worst_config,
            "worst_parameters": worst_params,
            "timestamp": datetime.now().isoformat()
        }
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"\nWorst configuration saved to {config_file}")
    
    # Create visualization of worst parameters (replaces old one)
    worst_plot_path = "visualizations/worst_parameters.png"
    visualizer.plot_worst_parameters(
        worst_params, 
        worst_config,
        save_path=worst_plot_path
    )
    
    # Generate comprehensive dashboard with all parameter sweep data
    print("\nGenerating comprehensive parameter sweep dashboard...")
    visualizer.plot_comprehensive_sweep_dashboard(
        sweep.results,
        save_path="visualizations/comprehensive_sweep_dashboard.html"
    )
    print("Comprehensive dashboard saved to visualizations/comprehensive_sweep_dashboard.html")
    
    print(f"\nTotal stance shift plots generated: {len(sweep.stance_plot_paths)}")
    
    return sweep.results, worst_params, worst_config

def run_baseline_comparison():
    """Run baseline configuration and compare with optimized parameters."""
    print("\n" + "="*60)
    print("Running Baseline Comparison")
    print("="*60)
    
    # Baseline: default parameters
    baseline_config = {
        "personalities": ["skeptic", "optimist"],
        "model": config.DEFAULT_MODEL,
        "max_turns": 5,
        "context_size": 1000
    }
    
    # Get best parameters from sweeps (would need to run sweeps first)
    # For now, use a reasonable configuration
    optimized_config = {
        "personalities": ["skeptic", "optimist", "persuader"],
        "model": config.DEFAULT_MODEL,
        "max_turns": 10,
        "context_size": 2000
    }
    
    # Test on a few claims
    test_claims = [
        get_claims_by_type(ClaimType.GROUND_TRUTH)[0],
        get_claims_by_type(ClaimType.FALSE)[0],
        get_claims_by_type(ClaimType.DEBATABLE)[0]
    ]
    
    judge = JudgeAgent()
    baseline_results = []
    optimized_results = []
    
    for claim in test_claims:
        # Baseline
        baseline_manager = ConversationManager(**baseline_config)
        baseline_result = baseline_manager.run_conversation(claim)
        baseline_eval = judge.evaluate_conversation(
            claim=baseline_result["claim"],
            claim_type=baseline_result["claim_type"],
            final_responses=baseline_result["final_responses"],
            conversation_history=baseline_result["conversation_history"]
        )
        baseline_results.append(baseline_eval)
        
        # Optimized
        optimized_manager = ConversationManager(**optimized_config)
        optimized_result = optimized_manager.run_conversation(claim)
        optimized_eval = judge.evaluate_conversation(
            claim=optimized_result["claim"],
            claim_type=optimized_result["claim_type"],
            final_responses=optimized_result["final_responses"],
            conversation_history=optimized_result["conversation_history"]
        )
        optimized_results.append(optimized_eval)
    
    # Compare results
    print("\nBaseline vs Optimized Comparison:")
    print("-" * 60)
    for i, claim in enumerate(test_claims):
        print(f"\nClaim: {claim.text[:50]}...")
        print(f"Baseline: {baseline_results[i].get('overall_correctness', 'unknown')}")
        print(f"Optimized: {optimized_results[i].get('overall_correctness', 'unknown')}")

def main():
    """Main function."""
    print("="*60)
    print("Agents Arguing Themselves into Hallucinations")
    print("="*60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\nAvailable modes:")
        print("1. single - Run a single conversation")
        print("2. sweep - Run parameter sweeps")
        print("3. baseline - Run baseline comparison")
        print("4. all - Run all modes")
        mode = input("\nEnter mode (single/sweep/baseline/all): ").strip().lower()
    
    if mode == "single":
        # Run a single conversation
        claim = get_claims_by_type(ClaimType.DEBATABLE)[0]
        run_single_conversation(claim)
    
    elif mode == "sweep":
        # Run parameter sweeps
        run_parameter_sweeps()
    
    elif mode == "baseline":
        # Run baseline comparison
        run_baseline_comparison()
    
    elif mode == "all":
        # Run everything
        run_single_conversation(get_claims_by_type(ClaimType.DEBATABLE)[0])
        run_parameter_sweeps()
        run_baseline_comparison()
    
    else:
        print(f"Unknown mode: {mode}")

if __name__ == "__main__":
    main()

