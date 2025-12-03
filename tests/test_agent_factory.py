"""
Tests for the agent factory module.
"""
import pytest
from agents.agent_factory import get_personality_config, get_available_personalities


class TestAgentFactory:
    """Tests for agent factory functions."""
    
    def test_get_available_personalities(self):
        """Test getting list of available personalities."""
        personalities = get_available_personalities()
        assert isinstance(personalities, list)
        assert len(personalities) > 0
        assert "skeptic" in personalities
        assert "optimist" in personalities
        assert "persuader" in personalities
        assert "deceiver" in personalities
    
    def test_get_personality_config_skeptic(self):
        """Test getting skeptic personality config."""
        config = get_personality_config("skeptic")
        assert isinstance(config, dict)
        assert "role" in config
        assert "goal" in config
        assert "backstory" in config
        assert "personality_traits" in config
        assert isinstance(config["personality_traits"], list)
    
    def test_get_personality_config_optimist(self):
        """Test getting optimist personality config."""
        config = get_personality_config("optimist")
        assert "role" in config
        assert "goal" in config
    
    def test_get_personality_config_persuader(self):
        """Test getting persuader personality config."""
        config = get_personality_config("persuader")
        assert "role" in config
        assert "goal" in config
    
    def test_get_personality_config_deceiver(self):
        """Test getting deceiver personality config."""
        config = get_personality_config("deceiver")
        assert "role" in config
        assert "goal" in config
    
    def test_get_personality_config_invalid(self):
        """Test getting invalid personality raises error."""
        with pytest.raises(ValueError) as exc_info:
            get_personality_config("invalid_personality")
        assert "Unknown personality" in str(exc_info.value)
    
    def test_personality_config_structure(self):
        """Test that all personality configs have required fields."""
        personalities = get_available_personalities()
        for personality in personalities:
            config = get_personality_config(personality)
            assert "role" in config
            assert "goal" in config
            assert "backstory" in config
            assert "personality_traits" in config
            assert isinstance(config["personality_traits"], list)
            assert len(config["personality_traits"]) > 0

