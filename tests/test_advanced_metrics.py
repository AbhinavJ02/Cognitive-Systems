"""
Tests for the advanced metrics module.
"""
import pytest
from evaluation.advanced_metrics import (
    calculate_agreement_score,
    calculate_persuasion_score,
    calculate_conversation_quality_metrics
)


class TestAgreementScore:
    """Tests for agreement score calculation."""
    
    def test_agreement_score_empty_tracking(self):
        """Test agreement score with empty stance tracking."""
        result = calculate_agreement_score({})
        assert result == 0.0
    
    def test_agreement_score_single_agent(self, sample_stance_tracking):
        """Test agreement score with single agent (should be 1.0)."""
        single_agent = {"skeptic": sample_stance_tracking["skeptic"]}
        result = calculate_agreement_score(single_agent)
        assert result == 1.0
    
    def test_agreement_score_agreement(self):
        """Test agreement score when agents agree."""
        stance_tracking = {
            "agent1": [{"turn": 1, "stance": "support"}],
            "agent2": [{"turn": 1, "stance": "support"}],
            "agent3": [{"turn": 1, "stance": "support"}]
        }
        result = calculate_agreement_score(stance_tracking)
        assert result == 1.0  # All agree
    
    def test_agreement_score_disagreement(self):
        """Test agreement score when agents disagree."""
        stance_tracking = {
            "agent1": [{"turn": 1, "stance": "support"}],
            "agent2": [{"turn": 1, "stance": "oppose"}],
            "agent3": [{"turn": 1, "stance": "oppose"}]
        }
        result = calculate_agreement_score(stance_tracking)
        assert 0.0 <= result < 1.0  # Some disagreement
    
    def test_agreement_score_specific_turn(self, sample_stance_tracking):
        """Test agreement score at specific turn."""
        result = calculate_agreement_score(sample_stance_tracking, turn=1)
        assert 0.0 <= result <= 1.0
    
    def test_agreement_score_neutral_stances(self):
        """Test agreement score with neutral stances."""
        stance_tracking = {
            "agent1": [{"turn": 1, "stance": "neutral"}],
            "agent2": [{"turn": 1, "stance": "neutral"}]
        }
        result = calculate_agreement_score(stance_tracking)
        assert result == 1.0  # Neutral stances should agree


class TestPersuasionScore:
    """Tests for persuasion score calculation."""
    
    def test_persuasion_score_missing_agents(self):
        """Test persuasion score with missing agents."""
        stance_tracking = {"agent1": [{"turn": 1, "stance": "support"}]}
        result = calculate_persuasion_score(stance_tracking, "agent1", "agent2")
        assert result == 0.0
    
    def test_persuasion_score_insufficient_data(self):
        """Test persuasion score with insufficient stance data."""
        stance_tracking = {
            "target": [{"turn": 1, "stance": "support"}],
            "influencer": []
        }
        result = calculate_persuasion_score(stance_tracking, "target", "influencer")
        assert result == 0.0
    
    def test_persuasion_score_positive_influence(self):
        """Test persuasion score with positive influence."""
        stance_tracking = {
            "target": [
                {"turn": 1, "stance": "oppose"},
                {"turn": 3, "stance": "support"}
            ],
            "influencer": [
                {"turn": 2, "stance": "support"}
            ]
        }
        result = calculate_persuasion_score(stance_tracking, "target", "influencer")
        assert result > 0.0  # Positive influence
    
    def test_persuasion_score_no_change(self):
        """Test persuasion score when target doesn't change."""
        stance_tracking = {
            "target": [
                {"turn": 1, "stance": "support"},
                {"turn": 3, "stance": "support"}
            ],
            "influencer": [
                {"turn": 2, "stance": "support"}
            ]
        }
        result = calculate_persuasion_score(stance_tracking, "target", "influencer")
        # No change means no persuasion
        assert result == 0.0


class TestConversationQualityMetrics:
    """Tests for conversation quality metrics calculation."""
    
    def test_calculate_quality_metrics_basic(self, sample_conversation_result, sample_evaluation):
        """Test basic quality metrics calculation."""
        metrics = calculate_conversation_quality_metrics(sample_conversation_result, sample_evaluation)
        
        assert "final_agreement_score" in metrics
        assert "average_agreement" in metrics
        assert "num_stance_flips" in metrics
        assert "average_confidence" in metrics
        assert isinstance(metrics["final_agreement_score"], float)
        assert isinstance(metrics["average_agreement"], float)
        assert isinstance(metrics["num_stance_flips"], int)
        assert isinstance(metrics["average_confidence"], float)
    
    def test_calculate_quality_metrics_ranges(self, sample_conversation_result, sample_evaluation):
        """Test that metrics are in valid ranges."""
        metrics = calculate_conversation_quality_metrics(sample_conversation_result, sample_evaluation)
        
        assert 0.0 <= metrics["final_agreement_score"] <= 1.0
        assert 0.0 <= metrics["average_agreement"] <= 1.0
        assert metrics["num_stance_flips"] >= 0
        assert 0.0 <= metrics["average_confidence"] <= 1.0
    
    def test_calculate_quality_metrics_empty_conversation(self, sample_evaluation):
        """Test quality metrics with empty conversation."""
        empty_result = {
            "claim": "Test",
            "claim_type": "ground_truth",
            "personalities": [],
            "conversation_history": [],
            "stance_tracking": {},
            "final_responses": {}
        }
        metrics = calculate_conversation_quality_metrics(empty_result, sample_evaluation)
        
        # Should handle empty gracefully
        assert "final_agreement_score" in metrics
        assert "average_agreement" in metrics

