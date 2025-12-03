"""
Tests for the dataset module.
"""
import pytest
from data.dataset import Claim, ClaimType, get_claims_by_type, get_all_claims, get_claim_statistics


class TestClaim:
    """Tests for the Claim class."""
    
    def test_claim_creation(self):
        """Test creating a claim."""
        claim = Claim("Test claim", ClaimType.GROUND_TRUTH)
        assert claim.text == "Test claim"
        assert claim.claim_type == ClaimType.GROUND_TRUTH
        assert claim.ground_truth == "Test claim"  # Defaults to text
    
    def test_claim_with_custom_ground_truth(self):
        """Test creating a claim with custom ground truth."""
        claim = Claim("Test claim", ClaimType.GROUND_TRUTH, "Custom ground truth")
        assert claim.ground_truth == "Custom ground truth"
    
    def test_claim_repr(self):
        """Test claim string representation."""
        claim = Claim("This is a very long claim that should be truncated in repr", ClaimType.DEBATABLE)
        repr_str = repr(claim)
        assert "Claim" in repr_str
        assert "debatable" in repr_str
        # The repr should contain the claim type and some text
        assert claim.claim_type.value in repr_str


class TestClaimType:
    """Tests for the ClaimType enum."""
    
    def test_claim_type_values(self):
        """Test that claim types have correct values."""
        assert ClaimType.GROUND_TRUTH.value == "ground_truth"
        assert ClaimType.FALSE.value == "false"
        assert ClaimType.DEBATABLE.value == "debatable"
    
    def test_all_claim_types_exist(self):
        """Test that all expected claim types exist."""
        expected_types = {ClaimType.GROUND_TRUTH, ClaimType.FALSE, ClaimType.DEBATABLE}
        assert len(expected_types) == 3


class TestDatasetFunctions:
    """Tests for dataset utility functions."""
    
    def test_get_claims_by_type_ground_truth(self):
        """Test getting ground truth claims."""
        claims = get_claims_by_type(ClaimType.GROUND_TRUTH)
        assert len(claims) > 0
        assert all(claim.claim_type == ClaimType.GROUND_TRUTH for claim in claims)
    
    def test_get_claims_by_type_false(self):
        """Test getting false claims."""
        claims = get_claims_by_type(ClaimType.FALSE)
        assert len(claims) > 0
        assert all(claim.claim_type == ClaimType.FALSE for claim in claims)
    
    def test_get_claims_by_type_debatable(self):
        """Test getting debatable claims."""
        claims = get_claims_by_type(ClaimType.DEBATABLE)
        assert len(claims) > 0
        assert all(claim.claim_type == ClaimType.DEBATABLE for claim in claims)
    
    def test_get_claims_by_type_invalid(self):
        """Test getting claims for invalid type returns empty list."""
        # Create a mock enum value that doesn't exist in dataset
        class MockClaimType:
            value = "nonexistent"
        claims = get_claims_by_type(MockClaimType())
        assert claims == []
    
    def test_get_all_claims(self):
        """Test getting all claims."""
        all_claims = get_all_claims()
        assert len(all_claims) > 0
        
        # Check that we have claims from all types
        claim_types = {claim.claim_type for claim in all_claims}
        assert ClaimType.GROUND_TRUTH in claim_types
        assert ClaimType.FALSE in claim_types
        assert ClaimType.DEBATABLE in claim_types
    
    def test_get_claim_statistics(self):
        """Test getting claim statistics."""
        stats = get_claim_statistics()
        assert "ground_truth" in stats
        assert "false" in stats
        assert "debatable" in stats
        assert all(isinstance(count, int) for count in stats.values())
        assert all(count > 0 for count in stats.values())
    
    def test_dataset_has_claims(self):
        """Test that dataset contains claims."""
        from data.dataset import DATASET
        assert len(DATASET) == 3  # Three claim types
        assert all(len(claims) > 0 for claims in DATASET.values())

