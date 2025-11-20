"""
Factory for creating agents with different personalities.
Note: This module provides personality configurations. The actual agent
interactions are handled directly via OpenAI API in conversation_manager.py.
"""
from typing import List, Dict
import config

def get_personality_config(personality: str) -> Dict:
    """
    Get personality configuration.
    
    Args:
        personality: One of 'skeptic', 'optimist', 'persuader', or 'deceiver'
    
    Returns:
        Dictionary with personality configuration
    """
    if personality not in config.AGENT_PERSONALITIES:
        raise ValueError(f"Unknown personality: {personality}. Must be one of {list(config.AGENT_PERSONALITIES.keys())}")
    
    return config.AGENT_PERSONALITIES[personality]

def get_available_personalities() -> List[str]:
    """Get list of available personality types."""
    return list(config.AGENT_PERSONALITIES.keys())

