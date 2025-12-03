"""
Tests for the judge agent module.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from judge.judge_agent import JudgeAgent, CorrectnessLevel


class TestCorrectnessLevel:
    """Tests for the CorrectnessLevel enum."""
    
    def test_correctness_level_values(self):
        """Test that correctness levels have correct values."""
        assert CorrectnessLevel.CORRECT.value == "correct"
        assert CorrectnessLevel.APPROXIMATE.value == "approximate"
        assert CorrectnessLevel.WRONG.value == "wrong"


class TestJudgeAgent:
    """Tests for the JudgeAgent class."""
    
    @pytest.fixture
    def judge(self):
        """Create a JudgeAgent instance for testing."""
        with patch('judge.judge_agent.OpenAI'):
            return JudgeAgent()
    
    def test_judge_initialization(self, judge):
        """Test judge initialization."""
        assert judge is not None
        assert judge.model is not None
        assert judge.temperature is not None
    
    def test_cosine_similarity_identical_vectors(self, judge):
        """Test cosine similarity with identical vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        similarity = judge._cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6  # Should be exactly 1.0
    
    def test_cosine_similarity_orthogonal_vectors(self, judge):
        """Test cosine similarity with orthogonal vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = judge._cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-6  # Should be exactly 0.0
    
    def test_cosine_similarity_opposite_vectors(self, judge):
        """Test cosine similarity with opposite vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        similarity = judge._cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-6  # Should be exactly -1.0
    
    def test_cosine_similarity_zero_vector(self, judge):
        """Test cosine similarity with zero vector."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        similarity = judge._cosine_similarity(vec1, vec2)
        assert similarity == 0.0
    
    def test_cosine_similarity_range(self, judge):
        """Test that cosine similarity is in valid range."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([4.0, 5.0, 6.0])
        similarity = judge._cosine_similarity(vec1, vec2)
        assert -1.0 <= similarity <= 1.0
    
    def test_verify_judge_accuracy_no_responses(self, judge):
        """Test accuracy verification with no responses."""
        result = judge.verify_judge_accuracy({}, {})
        assert result["overall_accuracy"] == 0.0
        assert result["accuracy_flag"] is False
        assert "No agent responses provided" in result["warnings"]
    
    @patch.object(JudgeAgent, '_get_embeddings')
    def test_verify_judge_accuracy_success(self, mock_embeddings, judge, sample_final_responses, sample_evaluation):
        """Test successful accuracy verification."""
        # Mock embeddings to return similar vectors (high similarity)
        mock_embeddings.return_value = np.array([
            [1.0, 0.0, 0.0],  # Actual responses embedding
            [0.95, 0.1, 0.0],  # Judge evaluation embedding (similar)
            [0.9, 0.1, 0.0],   # Agent 1 embedding
            [0.85, 0.15, 0.0], # Agent 1 judge text embedding
            [0.9, 0.1, 0.0],   # Agent 2 embedding
            [0.85, 0.15, 0.0], # Agent 2 judge text embedding
            [0.9, 0.1, 0.0],   # Agent 3 embedding
            [0.85, 0.15, 0.0], # Agent 3 judge text embedding
        ])
        
        result = judge.verify_judge_accuracy(sample_final_responses, sample_evaluation)
        
        assert "overall_accuracy" in result
        assert "per_agent_similarities" in result
        assert "accuracy_flag" in result
        assert "warnings" in result
        assert isinstance(result["overall_accuracy"], float)
        assert isinstance(result["per_agent_similarities"], dict)
        assert len(result["per_agent_similarities"]) == len(sample_final_responses)
    
    @patch.object(JudgeAgent, '_get_embeddings')
    def test_verify_judge_accuracy_low_similarity(self, mock_embeddings, judge, sample_final_responses, sample_evaluation):
        """Test accuracy verification with low similarity."""
        # Mock embeddings to return very different vectors (low similarity)
        mock_embeddings.return_value = np.array([
            [1.0, 0.0, 0.0],   # Actual responses embedding
            [0.0, 1.0, 0.0],   # Judge evaluation embedding (orthogonal - different)
            [1.0, 0.0, 0.0],   # Agent 1 embedding
            [0.0, 1.0, 0.0],   # Agent 1 judge text embedding (different)
        ])
        
        # Use only one agent for simplicity
        single_response = {"skeptic": sample_final_responses["skeptic"]}
        result = judge.verify_judge_accuracy(single_response, sample_evaluation, similarity_threshold=0.7)
        
        assert result["accuracy_flag"] is False
        assert len(result["warnings"]) > 0
    
    def test_classify_correctness_correct(self, judge):
        """Test correctness classification for correct."""
        level = judge.classify_correctness("correct")
        assert level == CorrectnessLevel.CORRECT
    
    def test_classify_correctness_approximate(self, judge):
        """Test correctness classification for approximate."""
        level = judge.classify_correctness("approximate")
        assert level == CorrectnessLevel.APPROXIMATE
    
    def test_classify_correctness_wrong(self, judge):
        """Test correctness classification for wrong."""
        level = judge.classify_correctness("wrong")
        assert level == CorrectnessLevel.WRONG
    
    def test_classify_correctness_case_insensitive(self, judge):
        """Test correctness classification is case insensitive."""
        assert judge.classify_correctness("CORRECT") == CorrectnessLevel.CORRECT
        assert judge.classify_correctness("Approximate") == CorrectnessLevel.APPROXIMATE
        assert judge.classify_correctness("WRONG") == CorrectnessLevel.WRONG
    
    def test_classify_correctness_unknown(self, judge):
        """Test correctness classification for unknown string."""
        level = judge.classify_correctness("unknown_string")
        # Should default to WRONG
        assert level == CorrectnessLevel.WRONG

