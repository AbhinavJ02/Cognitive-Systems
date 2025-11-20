"""
Example script demonstrating how to use the multi-agent conversation system.
"""
from conversation.conversation_manager import ConversationManager
from judge.judge_agent import JudgeAgent
from data.dataset import Claim, ClaimType, get_claims_by_type

def example_single_conversation():
    """Run a simple example conversation."""
    print("="*60)
    print("Example: Single Conversation")
    print("="*60)
    
    # Get a debatable claim
    claim = get_claims_by_type(ClaimType.DEBATABLE)[0]
    print(f"\nClaim: {claim.text}")
    print(f"Type: {claim.claim_type.value}\n")
    
    # Create conversation manager with personalities
    # Note: You can include "deceiver" to test how misinformation spreads
    manager = ConversationManager(
        personalities=["skeptic", "optimist", "persuader"],  # Add "deceiver" to test deception
        max_turns=6  # Short conversation for example
    )
    
    # Run the conversation
    result = manager.run_conversation(claim)
    
    # Evaluate with judge
    judge = JudgeAgent()
    evaluation = judge.evaluate_conversation(
        claim=result["claim"],
        claim_type=result["claim_type"],
        final_responses=result["final_responses"],
        conversation_history=result["conversation_history"]
    )
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Overall Correctness: {evaluation.get('overall_correctness', 'unknown')}")
    print(f"Converged: {evaluation.get('converged', False)}")
    print(f"Hallucinations Detected: {evaluation.get('hallucinations_detected', False)}")
    print(f"\nNumber of turns: {result['num_turns']}")
    print(f"Duration: {result['duration_seconds']:.2f} seconds")
    
    return result, evaluation

if __name__ == "__main__":
    example_single_conversation()

