"""
Dataset of claims for testing: ground truth facts, false statements, and debatable claims.
"""
from typing import List, Dict
from enum import Enum

class ClaimType(Enum):
    GROUND_TRUTH = "ground_truth"
    FALSE = "false"
    DEBATABLE = "debatable"

class Claim:
    def __init__(self, text: str, claim_type: ClaimType, ground_truth: str = None):
        self.text = text
        self.claim_type = claim_type
        self.ground_truth = ground_truth or text  # For ground truth, this is the correct answer
    
    def __repr__(self):
        return f"Claim(type={self.claim_type.value}, text='{self.text[:50]}...')"

# Dataset of 5-10 examples per category
DATASET = {
    ClaimType.GROUND_TRUTH: [
        Claim("The Earth orbits around the Sun, not the other way around.", ClaimType.GROUND_TRUTH),
        Claim("Water boils at 100 degrees Celsius at sea level under standard atmospheric pressure.", ClaimType.GROUND_TRUTH),
        Claim("The speed of light in a vacuum is approximately 299,792,458 meters per second.", ClaimType.GROUND_TRUTH),
        Claim("DNA stands for Deoxyribonucleic Acid.", ClaimType.GROUND_TRUTH),
        Claim("The human body has 206 bones in an adult skeleton.", ClaimType.GROUND_TRUTH),
        Claim("Photosynthesis converts carbon dioxide and water into glucose using sunlight.", ClaimType.GROUND_TRUTH),
        Claim("The Great Wall of China is visible from space with the naked eye.", ClaimType.GROUND_TRUTH),
        Claim("Sharks have been around for over 400 million years.", ClaimType.GROUND_TRUTH),
    ],
    ClaimType.FALSE: [
        Claim("Humans only use 10% of their brain capacity.", ClaimType.FALSE),
        Claim("Bats are blind.", ClaimType.FALSE),
        Claim("Goldfish have a 3-second memory.", ClaimType.FALSE),
        Claim("Shaving makes hair grow back thicker.", ClaimType.FALSE),
        Claim("Sugar makes children hyperactive.", ClaimType.FALSE),
        Claim("The Great Wall of China is the only human-made structure visible from space.", ClaimType.FALSE),
        Claim("Dogs see in black and white.", ClaimType.FALSE),
        Claim("You can catch a cold from being cold.", ClaimType.FALSE),
    ],
    ClaimType.DEBATABLE: [
        Claim("Artificial intelligence will replace most human jobs within the next 20 years.", ClaimType.DEBATABLE),
        Claim("Social media has a net positive impact on society.", ClaimType.DEBATABLE),
        Claim("Climate change is primarily caused by human activity.", ClaimType.DEBATABLE),
        Claim("Universal basic income would improve economic equality.", ClaimType.DEBATABLE),
        Claim("Remote work is more productive than office work.", ClaimType.DEBATABLE),
        Claim("Cryptocurrency will replace traditional currency.", ClaimType.DEBATABLE),
        Claim("Genetically modified foods are safe for human consumption.", ClaimType.DEBATABLE),
        Claim("The benefits of nuclear energy outweigh the risks.", ClaimType.DEBATABLE),
    ]
}

def get_claims_by_type(claim_type: ClaimType) -> List[Claim]:
    """Get all claims of a specific type."""
    return DATASET.get(claim_type, [])

def get_all_claims() -> List[Claim]:
    """Get all claims from all categories."""
    all_claims = []
    for claims in DATASET.values():
        all_claims.extend(claims)
    return all_claims

def get_claim_statistics() -> Dict[str, int]:
    """Get statistics about the dataset."""
    return {
        claim_type.value: len(claims) 
        for claim_type, claims in DATASET.items()
    }

