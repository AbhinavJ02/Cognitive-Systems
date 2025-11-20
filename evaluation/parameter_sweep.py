"""
Parameter sweep system for systematic evaluation.
Supports parallel execution for faster sweeps.
"""
from typing import List, Dict, Any
from conversation.conversation_manager import ConversationManager
from judge.judge_agent import JudgeAgent
from data.dataset import Claim, ClaimType, get_claims_by_type
from results.debate_storage import save_debate_summary
from visualization.visualizer import Visualizer
import json
import os
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing

def _run_single_sweep_task(task_config: Dict) -> Dict:
    """
    Helper function to run a single parameter sweep task.
    This is used for parallel execution.
    
    Args:
        task_config: Dictionary with task configuration including:
            - parameter: parameter name
            - parameter_value: value to test
            - claim: Claim object (or dict representation)
            - personalities: list of personalities (or None, may be in parameter_value)
            - model: model name
            - max_turns: max turns
            - context_size: context size
            - temperature: temperature
    
    Returns:
        Result dictionary
    """
    from data.dataset import Claim, ClaimType
    
    # Reconstruct claim if needed
    if isinstance(task_config["claim"], dict):
        claim = Claim(text=task_config["claim"]["text"], 
                     claim_type=ClaimType(task_config["claim"]["claim_type"]))
    else:
        claim = task_config["claim"]
    
    parameter = task_config["parameter"]
    parameter_value = task_config["parameter_value"]
    
    # Create conversation manager with appropriate parameters
    manager_kwargs = {
        "personalities": task_config.get("personalities"),
        "model": task_config.get("model"),
        "max_turns": task_config.get("max_turns"),
        "context_size": task_config.get("context_size"),
        "temperature": task_config.get("temperature")
    }
    
    # Set the parameter being swept
    if parameter == "context_size":
        manager_kwargs["context_size"] = parameter_value
    elif parameter == "max_turns":
        manager_kwargs["max_turns"] = parameter_value
    elif parameter == "temperature":
        manager_kwargs["temperature"] = parameter_value
    elif parameter == "personality_combinations":
        # For personality combinations, parameter_value IS the personalities list
        manager_kwargs["personalities"] = parameter_value
    
    manager = ConversationManager(**{k: v for k, v in manager_kwargs.items() if v is not None})
    conversation_result = manager.run_conversation(claim)
    
    judge = JudgeAgent()
    evaluation = judge.evaluate_conversation(
        claim=claim.text,
        claim_type=claim.claim_type.value,
        final_responses=conversation_result["final_responses"],
        conversation_history=conversation_result["conversation_history"]
    )
    
    # Save debate summary
    metadata = {
        "mode": "parameter_sweep",
        "sweep_type": parameter,
        "parameter_value": str(parameter_value) if parameter == "personality_combinations" else parameter_value,
        "claim": claim.text,
        "claim_type": claim.claim_type.value,
    }
    
    summary_path = save_debate_summary(
        conversation_result,
        evaluation,
        metadata=metadata,
    )
    
    result = {
        "parameter": parameter,
        "parameter_value": str(parameter_value) if parameter == "personality_combinations" else parameter_value,
        "claim": claim.text,
        "claim_type": claim.claim_type.value,
        "conversation_result": conversation_result,
        "evaluation": evaluation,
        "timestamp": datetime.now().isoformat(),
        "debate_summary_path": summary_path,
        "stance_plot_path": None
    }
    
    return result


class ParameterSweep:
    def __init__(self, results_dir: str = "results", generate_stance_plots: bool = True, 
                 parallel: bool = True, max_workers: int = None):
        """
        Initialize parameter sweep system.
        
        Args:
            results_dir: Directory to save results
            generate_stance_plots: Whether to generate stance shift plots for each debate
            parallel: Whether to use parallel execution
            max_workers: Maximum number of parallel workers (None for auto-detect)
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results: List[Dict] = []
        self.generate_stance_plots = generate_stance_plots
        self.visualizer = Visualizer() if generate_stance_plots else None
        self.stance_plot_paths: List[str] = []
        self.parallel = parallel
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 4)  # Limit to 4 for API rate limits
    
    def sweep_context_size(self, context_sizes: List[int], 
                          claim: Claim, personalities: List[str] = None,
                          model: str = None, max_turns: int = None,
                          temperature: float = None) -> List[Dict]:
        """Sweep over different context sizes."""
        results = []
        
        if self.parallel and len(context_sizes) > 1:
            # Parallel execution
            tasks = []
            for context_size in context_sizes:
                task_config = {
                    "parameter": "context_size",
                    "parameter_value": context_size,
                    "claim": {"text": claim.text, "claim_type": claim.claim_type.value},
                    "personalities": personalities,
                    "model": model,
                    "max_turns": max_turns,
                    "temperature": temperature
                }
                tasks.append(task_config)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(_run_single_sweep_task, task): task for task in tasks}
                
                with tqdm(total=len(tasks), desc="Sweeping context size (parallel)") as pbar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                            self.results.append(result)
                        except Exception as e:
                            print(f"Error in parallel task: {e}")
                        finally:
                            pbar.update(1)
        else:
            # Sequential execution
            for context_size in tqdm(context_sizes, desc="Sweeping context size"):
                manager = ConversationManager(
                    personalities=personalities,
                    model=model,
                    max_turns=max_turns,
                    context_size=context_size,
                    temperature=temperature
                )
                
                conversation_result = manager.run_conversation(claim)
                
                judge = JudgeAgent()
                evaluation = judge.evaluate_conversation(
                    claim=claim.text,
                    claim_type=claim.claim_type.value,
                    final_responses=conversation_result["final_responses"],
                    conversation_history=conversation_result["conversation_history"]
                )
                
                stance_plot_path = None
                
                metadata = {
                    "mode": "parameter_sweep",
                    "sweep_type": "context_size",
                    "parameter_value": context_size,
                    "claim": claim.text,
                    "claim_type": claim.claim_type.value,
                }
                summary_path = save_debate_summary(
                    conversation_result,
                    evaluation,
                    metadata=metadata,
                )
                
                result = {
                    "parameter": "context_size",
                    "parameter_value": context_size,
                    "claim": claim.text,
                    "claim_type": claim.claim_type.value,
                    "conversation_result": conversation_result,
                    "evaluation": evaluation,
                    "timestamp": datetime.now().isoformat(),
                    "debate_summary_path": summary_path,
                    "stance_plot_path": stance_plot_path
                }
                
                results.append(result)
                self.results.append(result)
        
        return results
    
    def sweep_max_turns(self, max_turns_list: List[int],
                       claim: Claim, personalities: List[str] = None,
                       model: str = None, context_size: int = None,
                       temperature: float = None) -> List[Dict]:
        """Sweep over different numbers of turns."""
        results = []
        
        if self.parallel and len(max_turns_list) > 1:
            # Parallel execution
            tasks = []
            for max_turns in max_turns_list:
                task_config = {
                    "parameter": "max_turns",
                    "parameter_value": max_turns,
                    "claim": {"text": claim.text, "claim_type": claim.claim_type.value},
                    "personalities": personalities,
                    "model": model,
                    "max_turns": max_turns,
                    "context_size": context_size,
                    "temperature": temperature
                }
                tasks.append(task_config)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(_run_single_sweep_task, task): task for task in tasks}
                
                with tqdm(total=len(tasks), desc="Sweeping max turns (parallel)") as pbar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                            self.results.append(result)
                        except Exception as e:
                            print(f"Error in parallel task: {e}")
                        finally:
                            pbar.update(1)
        else:
            # Sequential execution
            for max_turns in tqdm(max_turns_list, desc="Sweeping max turns"):
                manager = ConversationManager(
                    personalities=personalities,
                    model=model,
                    max_turns=max_turns,
                    context_size=context_size,
                    temperature=temperature
                )
                
                conversation_result = manager.run_conversation(claim)
                
                judge = JudgeAgent()
                evaluation = judge.evaluate_conversation(
                    claim=claim.text,
                    claim_type=claim.claim_type.value,
                    final_responses=conversation_result["final_responses"],
                    conversation_history=conversation_result["conversation_history"]
                )
                
                stance_plot_path = None
                
                metadata = {
                    "mode": "parameter_sweep",
                    "sweep_type": "max_turns",
                    "parameter_value": max_turns,
                    "claim": claim.text,
                    "claim_type": claim.claim_type.value,
                }
                summary_path = save_debate_summary(
                    conversation_result,
                    evaluation,
                    metadata=metadata,
                )
                
                result = {
                    "parameter": "max_turns",
                    "parameter_value": max_turns,
                    "claim": claim.text,
                    "claim_type": claim.claim_type.value,
                    "conversation_result": conversation_result,
                    "evaluation": evaluation,
                    "timestamp": datetime.now().isoformat(),
                    "debate_summary_path": summary_path,
                    "stance_plot_path": stance_plot_path
                }
                
                results.append(result)
                self.results.append(result)
        
        return results
    
    def sweep_personality_combinations(self, personality_combinations: List[List[str]],
                                      claim: Claim, model: str = None,
                                      max_turns: int = None, context_size: int = None,
                                      temperature: float = None) -> List[Dict]:
        """Sweep over different personality combinations."""
        results = []
        
        if self.parallel and len(personality_combinations) > 1:
            # Parallel execution
            tasks = []
            for personalities in personality_combinations:
                task_config = {
                    "parameter": "personality_combinations",
                    "parameter_value": personalities,
                    "claim": {"text": claim.text, "claim_type": claim.claim_type.value},
                    "personalities": None,  # Will be set from parameter_value
                    "model": model,
                    "max_turns": max_turns,
                    "context_size": context_size,
                    "temperature": temperature
                }
                tasks.append(task_config)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(_run_single_sweep_task, task): task for task in tasks}
                
                with tqdm(total=len(tasks), desc="Sweeping personality combinations (parallel)") as pbar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                            self.results.append(result)
                        except Exception as e:
                            print(f"Error in parallel task: {e}")
                        finally:
                            pbar.update(1)
        else:
            # Sequential execution
            for personalities in tqdm(personality_combinations, desc="Sweeping personality combinations"):
                manager = ConversationManager(
                    personalities=personalities,
                    model=model,
                    max_turns=max_turns,
                    context_size=context_size,
                    temperature=temperature
                )
                
                conversation_result = manager.run_conversation(claim)
                
                judge = JudgeAgent()
                evaluation = judge.evaluate_conversation(
                    claim=claim.text,
                    claim_type=claim.claim_type.value,
                    final_responses=conversation_result["final_responses"],
                    conversation_history=conversation_result["conversation_history"]
                )
                
                stance_plot_path = None
                
                metadata = {
                    "mode": "parameter_sweep",
                    "sweep_type": "personality_combinations",
                    "parameter_value": personalities,
                    "claim": claim.text,
                    "claim_type": claim.claim_type.value,
                }
                summary_path = save_debate_summary(
                    conversation_result,
                    evaluation,
                    metadata=metadata,
                )
                
                result = {
                    "parameter": "personality_combinations",
                    "parameter_value": str(personalities),
                    "claim": claim.text,
                    "claim_type": claim.claim_type.value,
                    "conversation_result": conversation_result,
                    "evaluation": evaluation,
                    "timestamp": datetime.now().isoformat(),
                    "debate_summary_path": summary_path,
                    "stance_plot_path": stance_plot_path
                }
                
                results.append(result)
                self.results.append(result)
        
        return results
    
    def sweep_claim_types(self, claims: List[Claim],
                         personalities: List[str] = None,
                         model: str = None, max_turns: int = None,
                         context_size: int = None, temperature: float = None) -> List[Dict]:
        """Sweep over different claim types."""
        results = []
        
        if self.parallel and len(claims) > 1:
            # Parallel execution
            tasks = []
            for claim in claims:
                task_config = {
                    "parameter": "claim_type",
                    "parameter_value": claim.claim_type.value,
                    "claim": {"text": claim.text, "claim_type": claim.claim_type.value},
                    "personalities": personalities,
                    "model": model,
                    "max_turns": max_turns,
                    "context_size": context_size,
                    "temperature": temperature
                }
                tasks.append(task_config)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(_run_single_sweep_task, task): task for task in tasks}
                
                with tqdm(total=len(tasks), desc="Sweeping claim types (parallel)") as pbar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                            self.results.append(result)
                        except Exception as e:
                            print(f"Error in parallel task: {e}")
                        finally:
                            pbar.update(1)
        else:
            # Sequential execution
            for claim in tqdm(claims, desc="Sweeping claim types"):
                manager = ConversationManager(
                    personalities=personalities,
                    model=model,
                    max_turns=max_turns,
                    context_size=context_size,
                    temperature=temperature
                )
                
                conversation_result = manager.run_conversation(claim)
                
                judge = JudgeAgent()
                evaluation = judge.evaluate_conversation(
                    claim=claim.text,
                    claim_type=claim.claim_type.value,
                    final_responses=conversation_result["final_responses"],
                    conversation_history=conversation_result["conversation_history"]
                )
                
                stance_plot_path = None
                
                metadata = {
                    "mode": "parameter_sweep",
                    "sweep_type": "claim_type",
                    "parameter_value": claim.claim_type.value,
                    "claim": claim.text,
                }
                summary_path = save_debate_summary(
                    conversation_result,
                    evaluation,
                    metadata=metadata,
                )
                
                result = {
                    "parameter": "claim_type",
                    "parameter_value": claim.claim_type.value,
                    "claim": claim.text,
                    "claim_type": claim.claim_type.value,
                    "conversation_result": conversation_result,
                    "evaluation": evaluation,
                    "timestamp": datetime.now().isoformat(),
                    "debate_summary_path": summary_path,
                    "stance_plot_path": stance_plot_path
                }
                
                results.append(result)
                self.results.append(result)
        
        return results
    
    def sweep_temperature(self, temperatures: List[float],
                         claim: Claim, personalities: List[str] = None,
                         model: str = None, max_turns: int = None, 
                         context_size: int = None) -> List[Dict]:
        """Sweep over different temperature values."""
        results = []
        
        if self.parallel and len(temperatures) > 1:
            # Parallel execution
            tasks = []
            for temperature in temperatures:
                task_config = {
                    "parameter": "temperature",
                    "parameter_value": temperature,
                    "claim": {"text": claim.text, "claim_type": claim.claim_type.value},
                    "personalities": personalities,
                    "model": model,
                    "max_turns": max_turns,
                    "context_size": context_size,
                    "temperature": temperature
                }
                tasks.append(task_config)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(_run_single_sweep_task, task): task for task in tasks}
                
                with tqdm(total=len(tasks), desc="Sweeping temperature (parallel)") as pbar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                            self.results.append(result)
                        except Exception as e:
                            print(f"Error in parallel task: {e}")
                        finally:
                            pbar.update(1)
        else:
            # Sequential execution
            for temperature in tqdm(temperatures, desc="Sweeping temperature"):
                manager = ConversationManager(
                    personalities=personalities,
                    model=model,
                    max_turns=max_turns,
                    context_size=context_size,
                    temperature=temperature
                )
                
                conversation_result = manager.run_conversation(claim)
                
                judge = JudgeAgent()
                evaluation = judge.evaluate_conversation(
                    claim=claim.text,
                    claim_type=claim.claim_type.value,
                    final_responses=conversation_result["final_responses"],
                    conversation_history=conversation_result["conversation_history"]
                )
                
                stance_plot_path = None
                
                metadata = {
                    "mode": "parameter_sweep",
                    "sweep_type": "temperature",
                    "parameter_value": temperature,
                    "claim": claim.text,
                    "claim_type": claim.claim_type.value,
                }
                summary_path = save_debate_summary(
                    conversation_result,
                    evaluation,
                    metadata=metadata,
                )
                
                result = {
                    "parameter": "temperature",
                    "parameter_value": temperature,
                    "claim": claim.text,
                    "claim_type": claim.claim_type.value,
                    "conversation_result": conversation_result,
                    "evaluation": evaluation,
                    "timestamp": datetime.now().isoformat(),
                    "debate_summary_path": summary_path,
                    "stance_plot_path": stance_plot_path
                }
                
                results.append(result)
                self.results.append(result)
        
        return results
    
    def save_results(self, filename: str = None):
        """Save all results to a JSON file."""
        if filename is None:
            filename = f"parameter_sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {filepath}")
        return filepath
    
    def get_worst_parameters(self) -> Dict[str, Any]:
        """Analyze results to find worst parameter values (most likely to cause hallucinations)."""
        if not self.results:
            return {}
        
        # Simple analysis: find parameters with lowest correctness rates / highest hallucination rates
        parameter_scores = {}
        
        for result in self.results:
            param = result["parameter"]
            param_value = result["parameter_value"]
            correctness = result["evaluation"].get("overall_correctness", "unknown")
            hallucinations = result["evaluation"].get("hallucinations_detected", False)
            
            if param not in parameter_scores:
                parameter_scores[param] = {}
            
            if param_value not in parameter_scores[param]:
                parameter_scores[param][param_value] = {
                    "correct": 0, "approximate": 0, "wrong": 0, 
                    "hallucinations": 0, "total": 0
                }
            
            parameter_scores[param][param_value]["total"] += 1
            if correctness in parameter_scores[param][param_value]:
                parameter_scores[param][param_value][correctness] += 1
            if hallucinations:
                parameter_scores[param][param_value]["hallucinations"] += 1
        
        # Find worst values (lowest correct rate, highest hallucination rate)
        worst_params = {}
        for param, values in parameter_scores.items():
            worst_value = None
            worst_score = -1  # Start low, we want maximum (worst)
            
            for value, scores in values.items():
                # Weight: wrong = 1.0, approximate = 0.5, correct = 0.0
                # Add hallucination rate (hallucinations/total)
                error_score = (
                    scores["wrong"] * 1.0 + 
                    scores["approximate"] * 0.5
                ) / scores["total"] if scores["total"] > 0 else 0
                hallucination_rate = scores["hallucinations"] / scores["total"] if scores["total"] > 0 else 0
                # Combined worst score: error rate + hallucination rate
                combined_score = error_score + hallucination_rate
                if combined_score > worst_score:
                    worst_score = combined_score
                    worst_value = value
            
            if worst_value:
                worst_params[param] = {
                    "value": worst_value,
                    "correct_rate": parameter_scores[param][worst_value]["correct"] / parameter_scores[param][worst_value]["total"] if parameter_scores[param][worst_value]["total"] > 0 else 0,
                    "error_rate": (parameter_scores[param][worst_value]["wrong"] + parameter_scores[param][worst_value]["approximate"] * 0.5) / parameter_scores[param][worst_value]["total"] if parameter_scores[param][worst_value]["total"] > 0 else 0,
                    "hallucination_rate": parameter_scores[param][worst_value]["hallucinations"] / parameter_scores[param][worst_value]["total"] if parameter_scores[param][worst_value]["total"] > 0 else 0,
                    "combined_score": worst_score,
                    "scores": parameter_scores[param][worst_value] if worst_value else {}
                }
        
        return worst_params
    
    def get_worst_config(self) -> Dict[str, Any]:
        """Get worst configuration based on parameters most likely to cause hallucinations."""
        worst_params = self.get_worst_parameters()
        
        if not worst_params:
            return {}
        
        worst_config = {}
        
        # Extract worst values
        if "context_size" in worst_params:
            worst_config["context_size"] = worst_params["context_size"]["value"]
        if "max_turns" in worst_params:
            worst_config["max_turns"] = worst_params["max_turns"]["value"]
        if "temperature" in worst_params:
            worst_config["temperature"] = worst_params["temperature"]["value"]
        if "personality_combinations" in worst_params:
            # Parse personality string back to list
            personalities_str = worst_params["personality_combinations"]["value"]
            if personalities_str and isinstance(personalities_str, str):
                # Remove brackets and quotes, split
                personalities_str = personalities_str.strip("[]'\"")
                worst_config["personalities"] = [p.strip().strip("'\"") for p in personalities_str.split(",")]
        
        return worst_config

