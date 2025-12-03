"""
Tests for the parameter sweep module.
"""
import pytest
import os
import tempfile
import shutil
from data.dataset import Claim, ClaimType

# Try to import ParameterSweep, skip tests if dependencies are missing
try:
    from evaluation.parameter_sweep import ParameterSweep
    PARAMETER_SWEEP_AVAILABLE = True
except ImportError:
    PARAMETER_SWEEP_AVAILABLE = False


@pytest.mark.skipif(not PARAMETER_SWEEP_AVAILABLE, reason="ParameterSweep dependencies not available")
class TestParameterSweep:
    """Tests for the ParameterSweep class."""
    
    @pytest.fixture
    def temp_results_dir(self):
        """Create a temporary directory for test results."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sweep(self, temp_results_dir):
        """Create a ParameterSweep instance for testing."""
        return ParameterSweep(results_dir=temp_results_dir, generate_stance_plots=False, parallel=False)
    
    @pytest.fixture
    def sample_claim(self):
        """Sample claim for testing."""
        return Claim("Test claim", ClaimType.GROUND_TRUTH)
    
    def test_parameter_sweep_initialization(self, sweep, temp_results_dir):
        """Test ParameterSweep initialization."""
        assert sweep.results_dir == temp_results_dir
        assert isinstance(sweep.results, list)
        assert len(sweep.results) == 0
        assert os.path.exists(temp_results_dir)
    
    def test_save_results(self, sweep, sample_claim):
        """Test saving results to file."""
        # Add a mock result
        sweep.results.append({
            "parameter": "test_param",
            "parameter_value": "test_value",
            "claim": sample_claim.text,
            "claim_type": sample_claim.claim_type.value,
            "evaluation": {"overall_correctness": "correct"}
        })
        
        filepath = sweep.save_results("test_results.json")
        assert os.path.exists(filepath)
        assert "test_results.json" in filepath
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def test_save_results_default_filename(self, sweep, sample_claim):
        """Test saving results with default filename."""
        sweep.results.append({
            "parameter": "test_param",
            "parameter_value": "test_value",
            "claim": sample_claim.text,
            "claim_type": sample_claim.claim_type.value,
            "evaluation": {"overall_correctness": "correct"}
        })
        
        filepath = sweep.save_results()
        assert os.path.exists(filepath)
        assert "parameter_sweep_results" in filepath
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def test_get_worst_parameters_empty(self, sweep):
        """Test getting worst parameters with no results."""
        worst = sweep.get_worst_parameters()
        assert worst == {}
    
    def test_get_worst_parameters_with_results(self, sweep, sample_claim):
        """Test getting worst parameters with results."""
        # Add results with different correctness levels
        sweep.results.extend([
            {
                "parameter": "context_size",
                "parameter_value": 1000,
                "claim": sample_claim.text,
                "claim_type": sample_claim.claim_type.value,
                "evaluation": {"overall_correctness": "wrong", "hallucinations_detected": True}
            },
            {
                "parameter": "context_size",
                "parameter_value": 2000,
                "claim": sample_claim.text,
                "claim_type": sample_claim.claim_type.value,
                "evaluation": {"overall_correctness": "correct", "hallucinations_detected": False}
            }
        ])
        
        worst = sweep.get_worst_parameters()
        assert "context_size" in worst
        assert worst["context_size"]["value"] == 1000  # Should be the worst one
    
    def test_get_worst_config_empty(self, sweep):
        """Test getting worst config with no results."""
        config = sweep.get_worst_config()
        assert config == {}
    
    def test_get_worst_config_with_results(self, sweep, sample_claim):
        """Test getting worst config with results."""
        sweep.results.extend([
            {
                "parameter": "context_size",
                "parameter_value": 1000,
                "claim": sample_claim.text,
                "claim_type": sample_claim.claim_type.value,
                "evaluation": {"overall_correctness": "wrong", "hallucinations_detected": True}
            },
            {
                "parameter": "max_turns",
                "parameter_value": 5,
                "claim": sample_claim.text,
                "claim_type": sample_claim.claim_type.value,
                "evaluation": {"overall_correctness": "wrong", "hallucinations_detected": True}
            }
        ])
        
        config = sweep.get_worst_config()
        # Should contain worst values
        assert "context_size" in config or "max_turns" in config

