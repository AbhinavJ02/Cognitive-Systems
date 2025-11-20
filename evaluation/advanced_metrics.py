"""
Advanced metrics for analyzing multi-agent conversations.
Includes agreement scores, persuasion tracking, and influence analysis.
"""
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np


def calculate_agreement_score(stance_tracking: Dict[str, List[Dict]], turn: int = None) -> float:
    """
    Calculate agreement score between agents at a specific turn or overall.
    
    Args:
        stance_tracking: Dictionary mapping personality to list of stance dictionaries
        turn: Specific turn to check (None for final positions)
    
    Returns:
        Agreement score between 0.0 (complete disagreement) and 1.0 (complete agreement)
    """
    if not stance_tracking:
        return 0.0
    
    # Get stances at specific turn or final stances
    stances = []
    for personality, stance_list in stance_tracking.items():
        if not stance_list:
            continue
        
        if turn is not None:
            # Find stance at or before this turn
            stance_at_turn = None
            for s in stance_list:
                if s["turn"] <= turn:
                    stance_at_turn = s
                else:
                    break
            if stance_at_turn:
                stances.append(stance_at_turn["stance"])
        else:
            # Use final stance
            stances.append(stance_list[-1]["stance"])
    
    if len(stances) < 2:
        return 1.0  # Only one agent, perfect agreement
    
    # Map stances to numerical values for comparison
    stance_map = {"support": 1, "oppose": -1, "uncertain": 0, "neutral": 0}
    stance_values = [stance_map.get(s.lower(), 0) for s in stances]
    
    # Calculate pairwise agreement
    agreements = 0
    total_pairs = 0
    
    for i in range(len(stance_values)):
        for j in range(i + 1, len(stance_values)):
            total_pairs += 1
            if stance_values[i] == stance_values[j]:
                agreements += 1
    
    return agreements / total_pairs if total_pairs > 0 else 0.0


def calculate_persuasion_score(stance_tracking: Dict[str, List[Dict]], 
                               target_agent: str, 
                               influencing_agent: str) -> float:
    """
    Calculate how much one agent persuaded another.
    
    Args:
        stance_tracking: Dictionary mapping personality to list of stance dictionaries
        target_agent: Agent whose stance we're tracking
        influencing_agent: Agent who might have influenced the target
    
    Returns:
        Persuasion score between -1.0 (strong negative influence) and 1.0 (strong positive influence)
    """
    if target_agent not in stance_tracking or influencing_agent not in stance_tracking:
        return 0.0
    
    target_stances = stance_tracking[target_agent]
    influencer_stances = stance_tracking[influencing_agent]
    
    if len(target_stances) < 2 or len(influencer_stances) < 1:
        return 0.0
    
    # Track stance changes after influencer spoke
    stance_map = {"support": 1, "oppose": -1, "uncertain": 0, "neutral": 0}
    
    initial_target_stance = stance_map.get(target_stances[0]["stance"].lower(), 0)
    final_target_stance = stance_map.get(target_stances[-1]["stance"].lower(), 0)
    
    # Calculate average influencer stance
    influencer_avg = np.mean([stance_map.get(s["stance"].lower(), 0) for s in influencer_stances])
    
    # If target moved toward influencer's position, positive persuasion
    if influencer_avg > 0 and final_target_stance > initial_target_stance:
        return abs(final_target_stance - initial_target_stance)
    elif influencer_avg < 0 and final_target_stance < initial_target_stance:
        return abs(final_target_stance - initial_target_stance)
    elif influencer_avg == 0:
        return 0.0
    else:
        # Moved away from influencer
        return -abs(final_target_stance - initial_target_stance)


def calculate_influence_matrix(stance_tracking: Dict[str, List[Dict]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate influence matrix showing how much each agent influences each other.
    
    Args:
        stance_tracking: Dictionary mapping personality to list of stance dictionaries
    
    Returns:
        Dictionary mapping (agent1, agent2) to influence score
    """
    agents = list(stance_tracking.keys())
    influence_matrix = {}
    
    for agent1 in agents:
        influence_matrix[agent1] = {}
        for agent2 in agents:
            if agent1 == agent2:
                influence_matrix[agent1][agent2] = 0.0
            else:
                influence_matrix[agent1][agent2] = calculate_persuasion_score(
                    stance_tracking, agent2, agent1
                )
    
    return influence_matrix


def detect_stance_flips(stance_tracking: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
    """
    Detect when agents flip their stance (e.g., support -> oppose).
    
    Args:
        stance_tracking: Dictionary mapping personality to list of stance dictionaries
    
    Returns:
        List of flip events with agent, turn, previous stance, new stance
    """
    flips = []
    
    stance_map = {"support": 1, "oppose": -1, "uncertain": 0, "neutral": 0}
    
    for personality, stance_list in stance_tracking.items():
        if len(stance_list) < 2:
            continue
        
        for i in range(1, len(stance_list)):
            prev_stance = stance_map.get(stance_list[i-1]["stance"].lower(), 0)
            curr_stance = stance_map.get(stance_list[i]["stance"].lower(), 0)
            
            # Detect significant stance changes (crossing support/oppose boundary)
            if (prev_stance > 0 and curr_stance < 0) or (prev_stance < 0 and curr_stance > 0):
                flips.append({
                    "agent": personality,
                    "turn": stance_list[i]["turn"],
                    "round": stance_list[i]["round"],
                    "previous_stance": stance_list[i-1]["stance"],
                    "new_stance": stance_list[i]["stance"],
                    "previous_sentiment": stance_list[i-1].get("sentiment", "neutral"),
                    "new_sentiment": stance_list[i].get("sentiment", "neutral")
                })
    
    return flips


def calculate_turn_by_turn_agreement(stance_tracking: Dict[str, List[Dict]], 
                                    max_turn: int = None) -> List[Tuple[int, float]]:
    """
    Calculate agreement score for each turn.
    
    Args:
        stance_tracking: Dictionary mapping personality to list of stance dictionaries
        max_turn: Maximum turn to analyze (None for all turns)
    
    Returns:
        List of (turn, agreement_score) tuples
    """
    if not stance_tracking:
        return []
    
    # Find maximum turn
    all_turns = []
    for stance_list in stance_tracking.values():
        for stance in stance_list:
            all_turns.append(stance["turn"])
    
    if not all_turns:
        return []
    
    max_turn_value = min(max(all_turns), max_turn) if max_turn else max(all_turns)
    
    turn_agreements = []
    for turn in range(1, max_turn_value + 1):
        agreement = calculate_agreement_score(stance_tracking, turn)
        turn_agreements.append((turn, agreement))
    
    return turn_agreements


def calculate_conversation_quality_metrics(conversation_result: Dict[str, Any],
                                          evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive quality metrics for a conversation.
    
    Args:
        conversation_result: Full conversation result dictionary
        evaluation: Judge evaluation dictionary
    
    Returns:
        Dictionary with various quality metrics
    """
    stance_tracking = conversation_result.get("stance_tracking", {})
    stance_summary = conversation_result.get("stance_summary", {})
    
    metrics = {
        "final_agreement_score": calculate_agreement_score(stance_tracking),
        "average_agreement": 0.0,
        "num_stance_flips": 0,
        "influence_matrix": {},
        "persuasion_events": [],
        "convergence_rate": 0.0,
        "hallucination_rate": 0.0,
        "average_confidence": 0.0
    }
    
    # Calculate average agreement across all turns
    turn_agreements = calculate_turn_by_turn_agreement(stance_tracking)
    if turn_agreements:
        metrics["average_agreement"] = np.mean([score for _, score in turn_agreements])
        metrics["agreement_over_time"] = turn_agreements
    
    # Count stance flips
    flips = detect_stance_flips(stance_tracking)
    metrics["num_stance_flips"] = len(flips)
    metrics["stance_flips"] = flips
    
    # Calculate influence matrix
    if len(stance_tracking) >= 2:
        metrics["influence_matrix"] = calculate_influence_matrix(stance_tracking)
    
    # Calculate convergence and hallucination rates
    metrics["converged"] = evaluation.get("converged", False)
    metrics["hallucinations_detected"] = evaluation.get("hallucinations_detected", False)
    
    # Calculate average confidence from stance tracking
    all_confidences = []
    for stance_list in stance_tracking.values():
        for stance in stance_list:
            if "confidence" in stance:
                all_confidences.append(stance["confidence"])
    
    if all_confidences:
        metrics["average_confidence"] = np.mean(all_confidences)
    
    # Calculate persuasion events (agents changing stances after others speak)
    for flip in flips:
        flip_turn = flip["turn"]
        # Find which agents spoke just before this flip
        influencing_agents = []
        for agent, stance_list in stance_tracking.items():
            if agent != flip["agent"]:
                for stance in stance_list:
                    if stance["turn"] < flip_turn and stance["turn"] >= flip_turn - 2:
                        influencing_agents.append(agent)
                        break
        
        if influencing_agents:
            metrics["persuasion_events"].append({
                "influenced_agent": flip["agent"],
                "influencing_agents": list(set(influencing_agents)),
                "turn": flip_turn,
                "previous_stance": flip["previous_stance"],
                "new_stance": flip["new_stance"]
            })
    
    return metrics

