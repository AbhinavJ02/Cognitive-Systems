"""
Judge agent for automated correctness classification.
"""
from openai import OpenAI
import config
from typing import Dict, List
from enum import Enum

class CorrectnessLevel(Enum):
    CORRECT = "correct"
    APPROXIMATE = "approximate"
    WRONG = "wrong"

class JudgeAgent:
    def __init__(self, model: str = None, temperature: float = None):
        """
        Initialize the judge agent.
        
        Args:
            model: OpenAI model to use for judging
            temperature: Temperature for the model (lower for consistency)
        """
        self.model = model or config.JUDGE_MODEL
        self.temperature = temperature or config.JUDGE_TEMPERATURE
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    def evaluate_conversation(self, claim: str, claim_type: str, 
                            final_responses: Dict[str, str],
                            conversation_history: List[Dict]) -> Dict:
        """
        Evaluate a conversation and classify correctness.
        
        Args:
            claim: The original claim
            claim_type: Type of claim (ground_truth, false, debatable)
            final_responses: Final responses from each agent
            conversation_history: Full conversation history
        
        Returns:
            Dictionary with evaluation results
        """
        # Build evaluation prompt
        responses_text = "\n\n".join([
            f"{agent}: {response}" 
            for agent, response in final_responses.items()
        ])
        
        # Get ground truth based on claim type
        if claim_type == "ground_truth":
            expected = "The claim should be evaluated as CORRECT"
        elif claim_type == "false":
            expected = "The claim should be evaluated as WRONG"
        else:  # debatable
            expected = "The claim is debatable and may have valid arguments on both sides"
        
        prompt = f"""You are an impartial judge evaluating a multi-agent conversation about a claim.

Original claim: "{claim}"
Claim type: {claim_type}
Expected evaluation: {expected}

Final responses from agents:
{responses_text}

Your task:
1. Evaluate whether the agents' final positions are correct, approximate, or wrong relative to the ground truth
2. Classify each agent's final stance
3. Determine if the group converged to the correct answer or drifted into misinformation
4. Identify any hallucinations or false information introduced

Respond with a JSON object in the following format:
{{
    "overall_correctness": "correct|approximate|wrong",
    "agent_evaluations": {{
        "agent_name": "correct|approximate|wrong"
    }},
    "converged": true|false,
    "hallucinations_detected": true|false,
    "reasoning": "brief explanation"
}}

Important: For agent_evaluations, include an entry for each agent in the conversation."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an impartial judge evaluating the correctness of multi-agent conversations. Be objective and factual. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            evaluation_text = response.choices[0].message.content.strip()
            
            # Parse structured JSON evaluation
            import json
            try:
                evaluation = json.loads(evaluation_text)
                # Ensure all required fields are present
                evaluation.setdefault("overall_correctness", "unknown")
                evaluation.setdefault("agent_evaluations", {})
                evaluation.setdefault("converged", False)
                evaluation.setdefault("hallucinations_detected", False)
                evaluation.setdefault("reasoning", "")
                evaluation["raw_evaluation"] = evaluation_text
                return evaluation
            except json.JSONDecodeError:
                # Fallback to old parsing if JSON is malformed
                evaluation = self._parse_evaluation(evaluation_text, final_responses)
                evaluation["raw_evaluation"] = evaluation_text
                return evaluation
            
        except Exception as e:
            return {
                "overall_correctness": "unknown",
                "agent_evaluations": {},
                "converged": False,
                "hallucinations_detected": False,
                "error": str(e),
                "raw_evaluation": f"Error during evaluation: {str(e)}"
            }
    
    def _parse_evaluation(self, evaluation_text: str, final_responses: Dict[str, str]) -> Dict:
        """Parse the judge's evaluation text into structured format."""
        evaluation = {
            "overall_correctness": "unknown",
            "agent_evaluations": {},
            "converged": False,
            "hallucinations_detected": False,
            "reasoning": evaluation_text
        }
        
        text_lower = evaluation_text.lower()
        
        # Extract overall correctness
        if "overall correctness:" in text_lower:
            for level in ["correct", "approximate", "wrong"]:
                if level in text_lower.split("overall correctness:")[1].split("\n")[0].lower():
                    evaluation["overall_correctness"] = level
                    break
        
        # Extract agent evaluations
        for agent in final_responses.keys():
            agent_lower = agent.lower()
            if agent_lower in text_lower:
                for level in ["correct", "approximate", "wrong"]:
                    if level in text_lower:
                        evaluation["agent_evaluations"][agent] = level
        
        # Extract convergence
        if "converged" in text_lower or "convergence: yes" in text_lower:
            evaluation["converged"] = True
        
        # Extract hallucinations
        if "hallucination" in text_lower and ("yes" in text_lower or "detected" in text_lower):
            evaluation["hallucinations_detected"] = True
        
        return evaluation
    
    def classify_correctness(self, correctness_str: str) -> CorrectnessLevel:
        """Convert correctness string to enum."""
        correctness_str = correctness_str.lower()
        if "correct" in correctness_str and "approximate" not in correctness_str:
            return CorrectnessLevel.CORRECT
        elif "approximate" in correctness_str:
            return CorrectnessLevel.APPROXIMATE
        else:
            return CorrectnessLevel.WRONG

