"""
Visualization tools for stance shifts and accuracy metrics.
Includes both static (matplotlib) and interactive (Plotly) visualizations.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import os

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Interactive visualizations disabled.")

class Visualizer:
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_stance_shifts(self, conversation_result: Dict, save_path: str = None):
        """
        Plot how each agent's stance changes throughout the conversation.
        
        Args:
            conversation_result: Result from ConversationManager.run_conversation
            save_path: Path to save the plot
        """
        stance_tracking = conversation_result["stance_tracking"]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color map for personalities (highlight deceiver)
        personality_colors = {
            "skeptic": "blue",
            "optimist": "green",
            "persuader": "purple",
            "deceiver": "red"  # Red to highlight deceptive agent
        }
        
        for personality, stances in stance_tracking.items():
            if not stances:
                continue
            
            turns = [s["turn"] for s in stances]
            # Convert sentiment to numeric for plotting
            sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
            sentiments = [sentiment_map.get(s["sentiment"], 0) for s in stances]
            
            color = personality_colors.get(personality.lower(), "gray")
            linewidth = 3 if personality.lower() == "deceiver" else 2  # Thicker line for deceiver
            
            ax.plot(turns, sentiments, marker='o', label=personality.capitalize(), 
                   linewidth=linewidth, color=color, markersize=8 if personality.lower() == "deceiver" else 6)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Turn", fontsize=12)
        ax.set_ylabel("Stance (Positive/Neutral/Negative)", fontsize=12)
        ax.set_title(f"Stance Shifts: {conversation_result['claim'][:50]}...", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_accuracy_vs_parameter(self, sweep_results: List[Dict], parameter_name: str, save_path: str = None):
        """
        Plot accuracy vs a parameter from sweep results.
        
        Args:
            sweep_results: List of results from parameter sweep
            parameter_name: Name of the parameter being swept
            save_path: Path to save the plot
        """
        # Extract data
        param_values = []
        correctness_counts = {"correct": [], "approximate": [], "wrong": []}
        
        for result in sweep_results:
            if result["parameter"] == parameter_name:
                param_values.append(result["parameter_value"])
                eval_result = result["evaluation"]
                correctness = eval_result.get("overall_correctness", "unknown")
                
                for key in correctness_counts:
                    correctness_counts[key].append(1 if correctness == key else 0)
        
        if not param_values:
            print(f"No results found for parameter: {parameter_name}")
            return
        
        # Create DataFrame
        df = pd.DataFrame({
            "parameter_value": param_values,
            "correct": correctness_counts["correct"],
            "approximate": correctness_counts["approximate"],
            "wrong": correctness_counts["wrong"]
        })
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(param_values))
        width = 0.25
        
        ax.bar([i - width for i in x], correctness_counts["correct"], width, 
               label="Correct", color='green', alpha=0.7)
        ax.bar(x, correctness_counts["approximate"], width, 
               label="Approximate", color='yellow', alpha=0.7)
        ax.bar([i + width for i in x], correctness_counts["wrong"], width, 
               label="Wrong", color='red', alpha=0.7)
        
        ax.set_xlabel(parameter_name.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Accuracy vs {parameter_name.replace('_', ' ').title()}", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in param_values], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_convergence_analysis(self, sweep_results: List[Dict], save_path: str = None):
        """
        Plot convergence analysis across different parameters.
        Creates separate heatmaps for each parameter type to avoid mixing issues.
        
        Args:
            sweep_results: List of results from parameter sweep
            save_path: Path to save the plot
        """
        # Extract convergence data
        convergence_data = []
        
        for result in sweep_results:
            eval_result = result["evaluation"]
            convergence_data.append({
                "parameter": result["parameter"],
                "parameter_value": str(result["parameter_value"]),
                "converged": eval_result.get("converged", False),
                "correctness": eval_result.get("overall_correctness", "unknown")
            })
        
        if not convergence_data:
            print("No convergence data found")
            return
        
        df = pd.DataFrame(convergence_data)
        
        # Get unique parameters
        unique_params = df["parameter"].unique()
        
        # Create separate heatmaps for each parameter
        n_params = len(unique_params)
        if n_params == 0:
            return
        
        # Calculate grid dimensions
        cols = min(2, n_params)
        rows = (n_params + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, param in enumerate(unique_params):
            param_data = df[df["parameter"] == param].copy()
            
            # Create pivot table for this parameter only
            pivot = param_data.pivot_table(
                values="converged",
                index="parameter",
                columns="parameter_value",
                aggfunc="mean"
            )
            
            # Sort columns if they're numeric
            try:
                # Try to sort by numeric value
                pivot.columns = pivot.columns.astype(float)
                pivot = pivot.sort_index(axis=1)
                pivot.columns = pivot.columns.astype(str)
            except (ValueError, TypeError):
                # If not numeric, sort alphabetically
                pivot = pivot.sort_index(axis=1)
            
            ax = axes[idx] if n_params > 1 else axes[0]
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", 
                       vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Convergence Rate'},
                       cbar=(idx == 0))  # Only show colorbar on first plot
            
            ax.set_title(f"Convergence Rate: {param.replace('_', ' ').title()}", fontsize=12)
            ax.set_xlabel("Parameter Value", fontsize=10)
            ax.set_ylabel("")
            ax.set_yticklabels([])
        
        # Hide unused subplots
        for idx in range(n_params, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle("Convergence Rate by Parameter", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_hallucination_detection(self, sweep_results: List[Dict], save_path: str = None):
        """
        Plot hallucination detection across different parameters.
        
        Args:
            sweep_results: List of results from parameter sweep
            save_path: Path to save the plot
        """
        # Extract hallucination data
        hallucination_data = []
        
        for result in sweep_results:
            eval_result = result.get("evaluation", {})
            # Get claim_type from result or conversation_result
            claim_type = result.get("claim_type")
            if not claim_type and "conversation_result" in result:
                claim_type = result["conversation_result"].get("claim_type", "unknown")
            
            hallucination_data.append({
                "parameter": result.get("parameter", "unknown"),
                "parameter_value": str(result.get("parameter_value", "")),
                "hallucinations": 1.0 if eval_result.get("hallucinations_detected", False) else 0.0,
                "claim_type": claim_type or "unknown"
            })
        
        if not hallucination_data:
            print("No hallucination data found")
            return
        
        df = pd.DataFrame(hallucination_data)
        
        # Get all unique parameters and claim types
        all_parameters = df["parameter"].unique()
        all_claim_types = df["claim_type"].unique()
        
        # Group by parameter and claim type and calculate mean hallucination rate
        grouped = (
            df.groupby(["parameter", "claim_type"])["hallucinations"]
            .mean()
            .reset_index()
        )
        
        # Create pivot table with all combinations
        pivot = grouped.pivot_table(
            index="parameter",
            columns="claim_type",
            values="hallucinations",
            fill_value=0.0,
        )
        
        # Ensure all parameters and claim types are present
        pivot = pivot.reindex(all_parameters, fill_value=0.0)
        for claim_type in all_claim_types:
            if claim_type not in pivot.columns:
                pivot[claim_type] = 0.0
        
        # Reorder columns to ensure consistent order
        claim_type_order = ["ground_truth", "false", "debatable"]
        existing_claim_types = [ct for ct in claim_type_order if ct in pivot.columns]
        remaining_claim_types = [ct for ct in pivot.columns if ct not in claim_type_order]
        pivot = pivot[existing_claim_types + remaining_claim_types]
        
        parameters = pivot.index.tolist()
        claim_types = pivot.columns.tolist()
        
        if not parameters or not claim_types:
            print("No valid parameters or claim types found for hallucination plot")
            return
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(parameters))
        n_claim_types = len(claim_types)
        width = 0.8 / max(n_claim_types, 1)  # distribute bars within each group
        
        # Color map for claim types
        colors = {
            "ground_truth": "green",
            "false": "orange", 
            "debatable": "blue"
        }
        
        for i, claim_type in enumerate(claim_types):
            values = pivot[claim_type].values
            offsets = (i - (n_claim_types - 1) / 2) * width
            color = colors.get(claim_type, "gray")
            
            ax.bar(
                x + offsets,
                values,
                width,
                label=claim_type.replace("_", " ").title(),
                align="center",
                color=color,
                alpha=0.7
            )
        
        ax.set_xlabel("Parameter", fontsize=12)
        ax.set_ylabel("Hallucination Rate", fontsize=12)
        ax.set_title("Hallucination Detection by Parameter and Claim Type", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(parameters, rotation=45, ha='right')
        ax.set_ylim(0, max(1.0, ax.get_ylim()[1] * 1.1))
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Hallucination detection plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_optimized_parameters(self, best_params: Dict[str, Any], optimized_config: Dict[str, Any], save_path: str = None):
        """
        Create a visualization showing optimized parameters and their performance.
        
        Args:
            best_params: Dictionary with best parameters and their scores
            optimized_config: Optimized configuration dictionary
            save_path: Path to save the plot
        """
        if not best_params:
            print("No parameters to visualize")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Parameter values
        ax1 = axes[0]
        params_list = []
        values_list = []
        correct_rates = []
        
        for param, info in best_params.items():
            if param == "personality_combinations":
                # Handle personality combinations specially
                param_display = "Personalities"
                value_display = str(info["value"]).replace("'", "").replace("[", "").replace("]", "")
            else:
                param_display = param.replace("_", " ").title()
                value_display = str(info["value"])
            
            params_list.append(param_display)
            values_list.append(value_display)
            correct_rates.append(info.get("correct_rate", 0))
        
        # Create bar plot for parameter values
        x_pos = range(len(params_list))
        ax1.barh(x_pos, correct_rates, color='steelblue', alpha=0.7)
        ax1.set_yticks(x_pos)
        ax1.set_yticklabels(params_list, fontsize=10)
        ax1.set_xlabel("Correctness Rate", fontsize=12)
        ax1.set_title("Optimized Parameters - Performance", fontsize=14)
        ax1.set_xlim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (rate, value) in enumerate(zip(correct_rates, values_list)):
            # Truncate long values
            display_value = value[:40] + "..." if len(value) > 40 else value
            ax1.text(rate + 0.01, i, f"{display_value} ({rate:.1%})", va='center', fontsize=9)
        
        # Right plot: Optimized configuration summary
        ax2 = axes[1]
        ax2.axis('off')
        
        config_text = "Optimized Configuration\n" + "="*40 + "\n\n"
        
        if optimized_config:
            if "context_size" in optimized_config:
                config_text += f"Context Size: {optimized_config['context_size']} tokens\n"
            if "max_turns" in optimized_config:
                config_text += f"Max Turns: {optimized_config['max_turns']}\n"
            if "personalities" in optimized_config:
                config_text += f"Personalities: {', '.join(optimized_config['personalities'])}\n"
        else:
            config_text += "No complete configuration available\n"
            config_text += "(Some parameters may not have been optimized)\n"
        
        config_text += "\n" + "Best Parameter Details\n" + "-"*40 + "\n\n"
        for param, info in best_params.items():
            param_name = param.replace("_", " ").title()
            config_text += f"{param_name}:\n"
            config_text += f"  Value: {info['value']}\n"
            config_text += f"  Correct Rate: {info.get('correct_rate', 0):.1%}\n"
            if 'weighted_score' in info:
                config_text += f"  Weighted Score: {info['weighted_score']:.2f}\n"
            config_text += "\n"
        
        ax2.text(0.1, 0.95, config_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle("Optimized Model Parameters", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Optimized parameters plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_optimized_parameters(self, best_params: Dict[str, Any], optimized_config: Dict[str, Any], save_path: str = None):
        """
        Create a visualization showing optimized parameters and their performance.
        """
        if not best_params:
            print("No parameters to visualize")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Parameter performance
        ax1 = axes[0]
        params_list = []
        values_list = []
        correct_rates = []
        
        for param, info in best_params.items():
            if param == "personality_combinations":
                param_display = "Personalities"
                value_display = str(info["value"]).replace("'", "").replace("[", "").replace("]", "")
            else:
                param_display = param.replace("_", " ").title()
                value_display = str(info["value"])
            
            params_list.append(param_display)
            values_list.append(value_display)
            correct_rates.append(info.get("correct_rate", 0))
        
        x_pos = range(len(params_list))
        ax1.barh(x_pos, correct_rates, color='steelblue', alpha=0.7)
        ax1.set_yticks(x_pos)
        ax1.set_yticklabels(params_list, fontsize=10)
        ax1.set_xlabel("Correctness Rate", fontsize=12)
        ax1.set_title("Optimized Parameters - Performance", fontsize=14)
        ax1.set_xlim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='x')
        
        for i, (rate, value) in enumerate(zip(correct_rates, values_list)):
            display_value = value[:40] + "..." if len(value) > 40 else value
            ax1.text(min(rate + 0.02, 1.02), i, f"{display_value} ({rate:.1%})", va='center', fontsize=9)
        
        # Right plot: Optimized configuration summary
        ax2 = axes[1]
        ax2.axis('off')
        
        config_text = "Optimized Configuration\n" + "="*40 + "\n\n"
        if optimized_config:
            if "context_size" in optimized_config:
                config_text += f"Context Size: {optimized_config['context_size']} tokens\n"
            if "max_turns" in optimized_config:
                config_text += f"Max Turns: {optimized_config['max_turns']}\n"
            if "personalities" in optimized_config:
                config_text += f"Personalities: {', '.join(optimized_config['personalities'])}\n"
        else:
            config_text += "No complete configuration available\n(Some parameters may not have been optimized)\n"
        
        config_text += "\nBest Parameter Details\n" + "-"*40 + "\n\n"
        for param, info in best_params.items():
            param_name = param.replace("_", " ").title()
            config_text += f"{param_name}:\n"
            config_text += f"  Value: {info['value']}\n"
            config_text += f"  Correct Rate: {info.get('correct_rate', 0):.1%}\n"
            if "weighted_score" in info:
                config_text += f"  Weighted Score: {info['weighted_score']:.2f}\n"
            config_text += "\n"
        
        ax2.text(
            0.05,
            0.95,
            config_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )
        
        plt.suptitle("Optimized Model Parameters", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = save_path or os.path.join(self.output_dir, "optimized_parameters.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Optimized parameters plot saved to {save_path}")
        plt.close()
    
    def plot_average_stance_shifts(self, sweep_results: List[Dict[str, Any]], parameter_name: str, save_dir: str = None):
        """
        Plot average stance shifts per parameter value by aggregating sentiments across runs.
        """
        relevant_results = [r for r in sweep_results if r.get("parameter") == parameter_name]
        if not relevant_results:
            print(f"No stance data available for parameter '{parameter_name}'")
            return
        
        sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
        save_dir = save_dir or self.output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Group results by parameter value
        grouped_results: Dict[Any, List[Dict[str, Any]]] = {}
        for result in relevant_results:
            value = result.get("parameter_value")
            grouped_results.setdefault(value, []).append(result)
        
        for value, results in grouped_results.items():
            aggregated: Dict[str, Dict[int, List[float]]] = {}
            for result in results:
                conversation = result.get("conversation_result", {})
                stance_tracking = conversation.get("stance_tracking", {})
                for personality, stances in stance_tracking.items():
                    personality_data = aggregated.setdefault(personality, {})
                    for stance in stances:
                        turn = stance.get("turn")
                        sentiment = stance.get("sentiment", "neutral")
                        score = sentiment_map.get(sentiment, 0)
                        personality_data.setdefault(turn, []).append(score)
            
            if not aggregated:
                continue
            
            fig, ax = plt.subplots(figsize=(12, 6))
            for personality, turn_scores in aggregated.items():
                turns_sorted = sorted(turn_scores.keys())
                avg_values = [np.mean(turn_scores[turn]) for turn in turns_sorted]
                ax.plot(
                    turns_sorted,
                    avg_values,
                    marker='o',
                    linewidth=2,
                    label=personality.capitalize()
                )
            
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylim(-1.1, 1.1)
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_xlabel("Turn", fontsize=12)
            ax.set_ylabel("Average Sentiment", fontsize=12)
            ax.set_title(f"Average Stance Shifts ({parameter_name.replace('_', ' ').title()} = {value})", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            value_slug = str(value).replace(" ", "_").replace("/", "-").replace("[", "").replace("]", "").replace("'", "")
            filename = f"stance_shifts_avg_{parameter_name}_{value_slug}.png"
            path = os.path.join(save_dir, filename)
            plt.tight_layout()
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Average stance shift plot saved to {path}")
            plt.close()
    
    def plot_worst_parameters(self, worst_params: Dict[str, Any], worst_config: Dict[str, Any], save_path: str = None):
        """
        Create a visualization showing worst parameters that lead to hallucinations.
        
        Args:
            worst_params: Dictionary with worst parameters and their scores
            worst_config: Worst configuration dictionary
            save_path: Path to save the plot
        """
        if not worst_params:
            print("No parameters to visualize")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Parameter error/hallucination rates
        ax1 = axes[0]
        params_list = []
        values_list = []
        hallucination_rates = []
        error_rates = []
        
        for param, info in worst_params.items():
            if param == "personality_combinations":
                param_display = "Personalities"
                value_display = str(info["value"]).replace("'", "").replace("[", "").replace("]", "")
            else:
                param_display = param.replace("_", " ").title()
                value_display = str(info["value"])
            
            params_list.append(param_display)
            values_list.append(value_display)
            hallucination_rates.append(info.get("hallucination_rate", 0))
            error_rates.append(info.get("error_rate", 0))
        
        # Create grouped bar chart
        x_pos = np.arange(len(params_list))
        width = 0.35
        
        bars1 = ax1.barh(x_pos - width/2, error_rates, width, label='Error Rate', color='red', alpha=0.7)
        bars2 = ax1.barh(x_pos + width/2, hallucination_rates, width, label='Hallucination Rate', color='orange', alpha=0.7)
        
        ax1.set_yticks(x_pos)
        ax1.set_yticklabels(params_list, fontsize=10)
        ax1.set_xlabel("Rate", fontsize=12)
        ax1.set_title("Worst Parameters - Error & Hallucination Rates", fontsize=14)
        ax1.set_xlim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.legend()
        
        # Add value labels
        for i, (err_rate, hall_rate, value) in enumerate(zip(error_rates, hallucination_rates, values_list)):
            display_value = value[:30] + "..." if len(value) > 30 else value
            ax1.text(max(err_rate, hall_rate) + 0.02, i, display_value, va='center', fontsize=8)
        
        # Right plot: Worst configuration summary
        ax2 = axes[1]
        ax2.axis('off')
        
        config_text = "Least Optimized Configuration\n" + "="*40 + "\n\n"
        config_text += "(Parameters Most Likely to Cause Hallucinations)\n\n"
        
        if worst_config:
            if "context_size" in worst_config:
                config_text += f"Context Size: {worst_config['context_size']} tokens\n"
            if "max_turns" in worst_config:
                config_text += f"Max Turns: {worst_config['max_turns']}\n"
            if "temperature" in worst_config:
                config_text += f"Temperature: {worst_config['temperature']}\n"
            if "personalities" in worst_config:
                config_text += f"Personalities: {', '.join(worst_config['personalities'])}\n"
        else:
            config_text += "No complete configuration available\n"
            config_text += "(Some parameters may not have been analyzed)\n"
        
        config_text += "\n" + "Worst Parameter Details\n" + "-"*40 + "\n\n"
        for param, info in worst_params.items():
            param_name = param.replace("_", " ").title()
            config_text += f"{param_name}:\n"
            config_text += f"  Value: {info['value']}\n"
            config_text += f"  Error Rate: {info.get('error_rate', 0):.1%}\n"
            config_text += f"  Hallucination Rate: {info.get('hallucination_rate', 0):.1%}\n"
            config_text += f"  Correct Rate: {info.get('correct_rate', 0):.1%}\n"
            if 'combined_score' in info:
                config_text += f"  Combined Score: {info['combined_score']:.2f}\n"
            config_text += "\n"
        
        ax2.text(0.1, 0.95, config_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        plt.suptitle("Parameters Most Likely to Cause Hallucinations", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Worst parameters plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    # ========== Interactive Plotly Visualizations ==========
    
    def plot_interactive_stance_shifts(self, conversation_result: Dict, save_path: str = None):
        """
        Create interactive Plotly visualization of stance shifts.
        
        Args:
            conversation_result: Result from ConversationManager.run_conversation
            save_path: Path to save HTML file (if None, shows in browser)
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Falling back to static visualization.")
            self.plot_stance_shifts(conversation_result, save_path)
            return
        
        stance_tracking = conversation_result["stance_tracking"]
        fig = go.Figure()
        
        # Color map for stances
        stance_colors = {
            "support": "green",
            "oppose": "red",
            "uncertain": "orange",
            "neutral": "gray"
        }
        
        # Color map for personalities (to highlight deceiver)
        personality_colors = {
            "skeptic": "blue",
            "optimist": "green",
            "persuader": "purple",
            "deceiver": "red"  # Red to highlight deceptive agent
        }
        
        for personality, stances in stance_tracking.items():
            if not stances:
                continue
            
            turns = [s["turn"] for s in stances]
            stance_values = [s["stance"] for s in stances]
            
            # Map stance to numeric for smooth lines
            stance_map = {"support": 1, "oppose": -1, "uncertain": 0, "neutral": 0}
            numeric_stances = [stance_map.get(s.lower(), 0) for s in stance_values]
            
            # Get color for personality (highlight deceiver in red)
            personality_colors = {
                "skeptic": "blue",
                "optimist": "green",
                "persuader": "purple",
                "deceiver": "red"
            }
            line_color = personality_colors.get(personality.lower(), "gray")
            line_width = 4 if personality.lower() == "deceiver" else 3  # Thicker line for deceiver
            
            fig.add_trace(go.Scatter(
                x=turns,
                y=numeric_stances,
                mode='lines+markers',
                name=personality.capitalize(),
                hovertemplate=f'<b>{personality.capitalize()}</b><br>' +
                            'Turn: %{x}<br>' +
                            'Stance: %{text}<br>' +
                            '<extra></extra>',
                text=stance_values,
                line=dict(width=line_width, color=line_color),
                marker=dict(size=10 if personality.lower() == "deceiver" else 8, color=line_color)
            ))
        
        fig.update_layout(
            title="Interactive Stance Shifts Over Time",
            xaxis_title="Turn",
            yaxis_title="Stance",
            yaxis=dict(
                tickmode='array',
                tickvals=[-1, 0, 1],
                ticktext=['Oppose', 'Neutral/Uncertain', 'Support'],
                range=[-1.2, 1.2]
            ),
            hovermode='closest',
            template='plotly_white',
            height=600
        )
        
        if save_path:
            if not save_path.endswith('.html'):
                save_path = save_path.replace('.png', '.html')
            fig.write_html(save_path)
            print(f"Interactive stance shifts plot saved to {save_path}")
        else:
            fig.show()
    
    def plot_interactive_agreement_tracking(self, conversation_result: Dict, save_path: str = None):
        """
        Create interactive plot showing agreement scores over time.
        
        Args:
            conversation_result: Result from ConversationManager.run_conversation
            save_path: Path to save HTML file
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Skipping interactive visualization.")
            return
        
        from evaluation.advanced_metrics import calculate_turn_by_turn_agreement
        
        stance_tracking = conversation_result.get("stance_tracking", {})
        turn_agreements = calculate_turn_by_turn_agreement(stance_tracking)
        
        if not turn_agreements:
            print("No agreement data available.")
            return
        
        turns, agreement_scores = zip(*turn_agreements)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=turns,
            y=agreement_scores,
            mode='lines+markers',
            name='Agreement Score',
            fill='tozeroy',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Agent Agreement Over Time",
            xaxis_title="Turn",
            yaxis_title="Agreement Score",
            yaxis=dict(range=[0, 1]),
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        if save_path:
            if not save_path.endswith('.html'):
                save_path = save_path.replace('.png', '.html')
            fig.write_html(save_path)
            print(f"Interactive agreement plot saved to {save_path}")
        else:
            fig.show()
    
    def plot_interactive_influence_matrix(self, conversation_result: Dict, save_path: str = None):
        """
        Create interactive heatmap showing influence between agents.
        
        Args:
            conversation_result: Result from ConversationManager.run_conversation
            save_path: Path to save HTML file
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Skipping interactive visualization.")
            return
        
        from evaluation.advanced_metrics import calculate_influence_matrix
        
        stance_tracking = conversation_result.get("stance_tracking", {})
        influence_matrix = calculate_influence_matrix(stance_tracking)
        
        if not influence_matrix:
            print("No influence data available.")
            return
        
        agents = list(influence_matrix.keys())
        influence_values = [[influence_matrix[agent1].get(agent2, 0) for agent2 in agents] for agent1 in agents]
        
        fig = go.Figure(data=go.Heatmap(
            z=influence_values,
            x=[a.capitalize() for a in agents],
            y=[a.capitalize() for a in agents],
            colorscale='RdYlGn',
            colorbar=dict(title="Influence Score"),
            text=[[f"{val:.2f}" for val in row] for row in influence_values],
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title="Agent Influence Matrix",
            xaxis_title="Influenced Agent",
            yaxis_title="Influencing Agent",
            height=600,
            template='plotly_white'
        )
        
        if save_path:
            if not save_path.endswith('.html'):
                save_path = save_path.replace('.png', '.html')
            fig.write_html(save_path)
            print(f"Interactive influence matrix saved to {save_path}")
        else:
            fig.show()
    
    def plot_interactive_metrics_dashboard(self, conversation_result: Dict, 
                                          evaluation: Dict, 
                                          advanced_metrics: Dict,
                                          save_path: str = None):
        """
        Create comprehensive interactive dashboard with multiple metrics.
        
        Args:
            conversation_result: Result from ConversationManager.run_conversation
            evaluation: Judge evaluation
            advanced_metrics: Advanced metrics dictionary
            save_path: Path to save HTML file
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Skipping interactive dashboard.")
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Stance Shifts Over Time',
                'Agreement Score Over Time',
                'Influence Matrix',
                'Quality Metrics'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "bar"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Stance shifts
        stance_tracking = conversation_result.get("stance_tracking", {})
        stance_map = {"support": 1, "oppose": -1, "uncertain": 0, "neutral": 0}
        personality_colors = {
            "skeptic": "blue",
            "optimist": "green",
            "persuader": "purple",
            "deceiver": "red"
        }
        
        for personality, stances in stance_tracking.items():
            if stances:
                turns = [s["turn"] for s in stances]
                numeric_stances = [stance_map.get(s["stance"].lower(), 0) for s in stances]
                color = personality_colors.get(personality.lower(), "gray")
                line_width = 4 if personality.lower() == "deceiver" else 3
                fig.add_trace(
                    go.Scatter(x=turns, y=numeric_stances, mode='lines+markers', 
                             name=personality.capitalize(),
                             line=dict(width=line_width, color=color),
                             marker=dict(size=10 if personality.lower() == "deceiver" else 8, color=color)),
                    row=1, col=1
                )
        
        # 2. Agreement over time
        if "agreement_over_time" in advanced_metrics:
            turn_agreements = advanced_metrics["agreement_over_time"]
            turns, scores = zip(*turn_agreements) if turn_agreements else ([], [])
            fig.add_trace(
                go.Scatter(x=turns, y=scores, mode='lines+markers', name='Agreement', fill='tozeroy'),
                row=1, col=2
            )
        
        # 3. Influence matrix
        if "influence_matrix" in advanced_metrics:
            influence_matrix = advanced_metrics["influence_matrix"]
            agents = list(influence_matrix.keys())
            influence_values = [[influence_matrix[agent1].get(agent2, 0) for agent2 in agents] 
                              for agent1 in agents]
            fig.add_trace(
                go.Heatmap(z=influence_values, x=[a.capitalize() for a in agents],
                          y=[a.capitalize() for a in agents], colorscale='RdYlGn'),
                row=2, col=1
            )
        
        # 4. Quality metrics bar chart
        metrics_names = ['Agreement', 'Confidence', 'Convergence']
        metrics_values = [
            advanced_metrics.get('final_agreement_score', 0),
            advanced_metrics.get('average_confidence', 0),
            1.0 if evaluation.get('converged', False) else 0.0
        ]
        fig.add_trace(
            go.Bar(x=metrics_names, y=metrics_values, name='Metrics'),
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="Turn", row=1, col=1)
        fig.update_yaxes(title_text="Stance", tickvals=[-1, 0, 1], 
                        ticktext=['Oppose', 'Neutral', 'Support'], row=1, col=1)
        fig.update_xaxes(title_text="Turn", row=1, col=2)
        fig.update_yaxes(title_text="Agreement Score", range=[0, 1], row=1, col=2)
        fig.update_xaxes(title_text="Influenced", row=2, col=1)
        fig.update_yaxes(title_text="Influencer", row=2, col=1)
        fig.update_xaxes(title_text="Metric", row=2, col=2)
        fig.update_yaxes(title_text="Score", range=[0, 1], row=2, col=2)
        
        fig.update_layout(
            height=900,
            title_text="Interactive Conversation Analysis Dashboard",
            showlegend=True,
            template='plotly_white'
        )
        
        if save_path:
            if not save_path.endswith('.html'):
                save_path = save_path.replace('.png', '.html')
            fig.write_html(save_path)
            print(f"Interactive dashboard saved to {save_path}")
        else:
            fig.show()
    
    def plot_comprehensive_sweep_dashboard(self, sweep_results: List[Dict], save_path: str = None):
        """
        Create a comprehensive interactive dashboard aggregating all parameter sweep results.
        Uses averages and organizes by parameter combinations with organized subplots.
        
        Args:
            sweep_results: List of all results from parameter sweeps
            save_path: Path to save HTML file
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Skipping comprehensive dashboard.")
            return
        
        if not sweep_results:
            print("No sweep results available for comprehensive dashboard.")
            return
        
        from evaluation.advanced_metrics import calculate_conversation_quality_metrics
        
        # Extract and aggregate data
        aggregated_data = []
        
        for result in sweep_results:
            conv = result.get("conversation_result", {})
            eval_result = result.get("evaluation", {})
            
            if not conv or not eval_result:
                continue
            
            # Calculate advanced metrics
            try:
                metrics = calculate_conversation_quality_metrics(conv, eval_result)
            except:
                metrics = {}
            
            # Extract parameter information
            parameter = result.get("parameter", "unknown")
            parameter_value = result.get("parameter_value", "unknown")
            claim_type = result.get("claim_type", "unknown")
            
            aggregated_data.append({
                "parameter": parameter,
                "parameter_value": str(parameter_value),
                "claim_type": claim_type,
                "overall_correctness": eval_result.get("overall_correctness", "unknown"),
                "converged": 1.0 if eval_result.get("converged", False) else 0.0,
                "hallucinations_detected": 1.0 if eval_result.get("hallucinations_detected", False) else 0.0,
                "agreement_score": metrics.get("final_agreement_score", 0.0),
                "average_agreement": metrics.get("average_agreement", 0.0),
                "num_stance_flips": metrics.get("num_stance_flips", 0),
                "average_confidence": metrics.get("average_confidence", 0.0),
            })
        
        if not aggregated_data:
            print("No valid data found for comprehensive dashboard.")
            return
        
        df = pd.DataFrame(aggregated_data)
        
        # Get unique parameters
        unique_parameters = sorted(df["parameter"].unique())
        num_params = len(unique_parameters)
        
        # Create organized subplot layout
        # Each parameter gets 3 rows: Convergence, Hallucination, Agreement
        # Use 2 columns for parameters
        cols = 2
        param_rows = (num_params + cols - 1) // cols
        total_rows = param_rows * 3  # 3 metrics per parameter
        
        # Color map for claim types
        claim_type_colors = {
            "ground_truth": "#2ecc71",  # Green
            "false": "#e74c3c",  # Red
            "debatable": "#3498db"  # Blue
        }
        
        # Create subplot titles
        subplot_titles = []
        for param in unique_parameters:
            param_title = param.replace('_', ' ').title()
            subplot_titles.extend([
                f"{param_title} - Convergence",
                f"{param_title} - Hallucination Rate",
                f"{param_title} - Agreement Score"
            ])
        
        # Create figure with subplots - 3 rows per parameter (convergence, hallucination, agreement)
        fig = make_subplots(
            rows=total_rows, cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.12,
            specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(total_rows)]
        )
        
        # Add traces for each parameter - 3 metrics per parameter
        for idx, param in enumerate(unique_parameters):
            param_data = df[df["parameter"] == param]
            
            # Group by parameter_value and claim_type, calculate averages
            grouped = param_data.groupby(["parameter_value", "claim_type"]).agg({
                "converged": "mean",
                "hallucinations_detected": "mean",
                "agreement_score": "mean",
                "average_agreement": "mean"
            }).reset_index()
            
            # Get sorted parameter values
            param_values = sorted(grouped["parameter_value"].unique(), key=str)
            x_pos = np.arange(len(param_values))
            
            # Claim types
            claim_types_sorted = sorted(grouped["claim_type"].unique())
            num_claim_types = len(claim_types_sorted)
            bar_width = 0.8 / max(num_claim_types, 1)
            
            # Calculate row positions for this parameter (3 metrics: convergence, hallucination, agreement)
            param_row_base = (idx // cols) * 3  # Base row for this parameter
            param_col = (idx % cols) + 1
            
            # Prepare data for all claim types
            all_data = {}
            for claim_type in claim_types_sorted:
                claim_data = grouped[grouped["claim_type"] == claim_type]
                convergence_values = []
                hallucination_values = []
                agreement_values = []
                
                for pv in param_values:
                    pv_data = claim_data[claim_data["parameter_value"] == str(pv)]
                    if not pv_data.empty:
                        convergence_values.append(pv_data["converged"].iloc[0])
                        hallucination_values.append(pv_data["hallucinations_detected"].iloc[0])
                        agreement_values.append(pv_data["agreement_score"].iloc[0])
                    else:
                        convergence_values.append(0.0)
                        hallucination_values.append(0.0)
                        agreement_values.append(0.0)
                
                all_data[claim_type] = {
                    "convergence": convergence_values,
                    "hallucination": hallucination_values,
                    "agreement": agreement_values
                }
            
            # Add traces for each metric
            for metric_idx, (metric_name, metric_key) in enumerate([
                ("Convergence", "convergence"),
                ("Hallucination Rate", "hallucination"),
                ("Agreement Score", "agreement")
            ]):
                row = param_row_base + metric_idx + 1
                
                for i, claim_type in enumerate(claim_types_sorted):
                    color = claim_type_colors.get(claim_type, "#95a5a6")
                    claim_display = claim_type.replace("_", " ").title()
                    offset = (i - (num_claim_types - 1) / 2) * bar_width
                    values = all_data[claim_type][metric_key]
                    
                    fig.add_trace(
                        go.Bar(
                            x=x_pos + offset,
                            y=values,
                            name=f"{claim_display}",
                            marker_color=color,
                            opacity=0.8,
                            legendgroup=claim_type,
                            showlegend=(idx == 0 and metric_idx == 0),
                            legendgrouptitle_text="Claim Type" if idx == 0 and metric_idx == 0 and i == 0 else None,
                            hovertemplate=f"<b>{param.replace('_', ' ').title()} - {metric_name}</b><br>" +
                                        f"Parameter Value: %{{customdata}}<br>" +
                                        f"{metric_name}: %{{y:.2%}}<br>" +
                                        f"Claim Type: {claim_display}<extra></extra>",
                            customdata=[str(pv) for pv in param_values]
                        ),
                        row=row, col=param_col
                    )
                
                # Update axes for this metric
                display_values = []
                for pv in param_values:
                    pv_str = str(pv)
                    if len(pv_str) > 12:
                        display_values.append(pv_str[:9] + "...")
                    else:
                        display_values.append(pv_str)
                
                fig.update_xaxes(
                    tickmode='array',
                    tickvals=x_pos,
                    ticktext=display_values if metric_idx == 2 else [],  # Only show labels on bottom row
                    tickangle=-45,
                    title_text="Parameter Value" if metric_idx == 2 else "",
                    row=row, col=param_col
                )
                fig.update_yaxes(
                    title_text=metric_name,
                    range=[0, 1.1],
                    row=row, col=param_col
                )
        
        # Update overall layout
        fig.update_layout(
            title={
                'text': "Comprehensive Parameter Sweep Dashboard<br><sub>Average Metrics (Convergence, Hallucination, Agreement) Across All Parameter Combinations</sub>",
                'x': 0.5,
                'xanchor': 'center'
            },
            height=250 * total_rows,  # Adjust height based on total rows
            barmode='group',
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.02 - (0.01 * param_rows),  # Adjust based on number of parameter rows
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            ),
            margin=dict(b=80, t=120, l=60, r=60)  # Extra margin for labels and title
        )
        
        if save_path:
            if not save_path.endswith('.html'):
                save_path = save_path.replace('.png', '.html')
            fig.write_html(save_path)
            print(f"Comprehensive sweep dashboard saved to {save_path}")
        else:
            fig.show()
    