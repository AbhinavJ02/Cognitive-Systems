"""
Pytest configuration and shared fixtures.
"""
import pytest
import numpy as np
from data.dataset import Claim, ClaimType
from typing import Dict, List


@pytest.fixture
def sample_ground_truth_claim():
    """Sample ground truth claim for testing."""
    return Claim(
        "The Earth orbits around the Sun.",
        ClaimType.GROUND_TRUTH
    )


@pytest.fixture
def sample_false_claim():
    """Sample false claim for testing."""
    return Claim(
        "Humans only use 10% of their brain.",
        ClaimType.FALSE
    )


@pytest.fixture
def sample_debatable_claim():
    """Sample debatable claim for testing."""
    return Claim(
        "Artificial intelligence will replace most jobs.",
        ClaimType.DEBATABLE
    )


@pytest.fixture
def sample_stance_tracking():
    """Sample stance tracking data for testing."""
    return {
        "skeptic": [
            {
                "turn": 1,
                "round": 1,
                "message": "I need evidence for this claim.",
                "stance": "oppose",
                "sentiment": "negative",
                "confidence": 0.8,
                "personality": "skeptic",
                "timestamp": "2024-01-01T00:00:00"
            },
            {
                "turn": 3,
                "round": 2,
                "message": "After reviewing the evidence, I agree.",
                "stance": "support",
                "sentiment": "positive",
                "confidence": 0.7,
                "personality": "skeptic",
                "timestamp": "2024-01-01T00:01:00"
            }
        ],
        "optimist": [
            {
                "turn": 2,
                "round": 1,
                "message": "This sounds reasonable to me.",
                "stance": "support",
                "sentiment": "positive",
                "confidence": 0.6,
                "personality": "optimist",
                "timestamp": "2024-01-01T00:00:30"
            },
            {
                "turn": 4,
                "round": 2,
                "message": "I still agree with this.",
                "stance": "support",
                "sentiment": "positive",
                "confidence": 0.7,
                "personality": "optimist",
                "timestamp": "2024-01-01T00:01:30"
            }
        ]
    }


@pytest.fixture
def sample_final_responses():
    """Sample final responses from agents."""
    return {
        "skeptic": "After careful consideration, I agree with the claim.",
        "optimist": "I fully support this claim.",
        "persuader": "This is absolutely correct and everyone should agree."
    }


@pytest.fixture
def sample_evaluation():
    """Sample judge evaluation."""
    return {
        "overall_correctness": "correct",
        "agent_evaluations": {
            "skeptic": "correct",
            "optimist": "correct",
            "persuader": "correct"
        },
        "converged": True,
        "hallucinations_detected": False,
        "reasoning": "All agents correctly evaluated the claim.",
        "raw_evaluation": '{"overall_correctness": "correct", "agent_evaluations": {"skeptic": "correct"}}'
    }


@pytest.fixture
def sample_conversation_result(sample_ground_truth_claim, sample_stance_tracking):
    """Sample conversation result."""
    return {
        "claim": sample_ground_truth_claim.text,
        "claim_type": sample_ground_truth_claim.claim_type.value,
        "personalities": ["skeptic", "optimist"],
        "conversation_history": [
            {
                "turn": 1,
                "agent": "skeptic",
                "message": "I need evidence for this claim.",
                "timestamp": "2024-01-01T00:00:00"
            },
            {
                "turn": 2,
                "agent": "optimist",
                "message": "This sounds reasonable to me.",
                "timestamp": "2024-01-01T00:00:30"
            }
        ],
        "stance_tracking": sample_stance_tracking,
        "final_responses": {
            "skeptic": "After careful consideration, I agree.",
            "optimist": "I fully support this claim."
        },
        "num_turns": 2,
        "duration_seconds": 60.0,
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_turns": 10,
        "context_size": 2000
    }


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing cosine similarity."""
    # Create simple mock embeddings (3-dimensional for testing)
    return {
        "text1": np.array([1.0, 0.0, 0.0]),
        "text2": np.array([0.0, 1.0, 0.0]),
        "text3": np.array([1.0, 1.0, 0.0]),
        "similar": np.array([0.9, 0.1, 0.0]),  # Similar to text1
    }

