"""
Example script demonstrating the deceiver agent in action.
This shows how a deceptive agent can influence other agents.
"""
from conversation.conversation_manager import ConversationManager
from judge.judge_agent import JudgeAgent
from data.dataset import Claim, ClaimType, get_claims_by_type

def example_with_deceiver():
    """Run a conversation with a deceiver agent to observe misinformation spread."""
    print("="*60)
    print("Example: Conversation with Deceiver Agent")
    print("="*60)
    
    # Use a ground truth claim to see if deceiver can make others doubt it
    claim = get_claims_by_type(ClaimType.GROUND_TRUTH)[0]
    print(f"\nClaim: {claim.text}")
    print(f"Type: {claim.claim_type.value} (This is a FACTUALLY CORRECT statement)")
    print("\n⚠️  Including a DECEIVER agent who will try to mislead others\n")
    
    # Create conversation manager with deceiver included
    manager = ConversationManager(
        personalities=["skeptic", "optimist", "deceiver"],
        max_turns=8  # More turns to see deception unfold
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
    
    # Show stance changes
    print("\n" + "="*60)
    print("Stance Summary")
    print("="*60)
    if "stance_summary" in result:
        for agent, summary in result["stance_summary"].items():
            print(f"\n{agent.upper()}:")
            print(f"  Initial stance: {summary.get('initial_stance', 'N/A')}")
            print(f"  Final stance: {summary.get('final_stance', 'N/A')}")
            print(f"  Positive responses: {summary.get('positive_count', 0)}")
            print(f"  Negative responses: {summary.get('negative_count', 0)}")
    
    print("\n" + "="*60)
    print("⚠️  Note: The deceiver agent deliberately tries to mislead others.")
    print("This is for research purposes to study how misinformation spreads.")
    print("="*60)
    
    return result, evaluation

if __name__ == "__main__":
    example_with_deceiver()

