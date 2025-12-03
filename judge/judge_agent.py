"""
Judge agent for automated correctness classification.
"""
from openai import OpenAI
import config
from typing import Dict, List
from enum import Enum
import numpy as np

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
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts using OpenAI's embedding model.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",  # or "text-embedding-ada-002"
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        except Exception as e:
            raise Exception(f"Failed to get embeddings: {str(e)}")
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def verify_judge_accuracy(self, final_responses: Dict[str, str], 
                              evaluation: Dict, 
                              similarity_threshold: float = 0.7) -> Dict:
        """
        Verify that the judge's evaluation accurately reflects the actual agent responses
        using cosine similarity of embeddings.
        
        Args:
            final_responses: Dictionary mapping agent names to their actual final responses
            evaluation: The judge's evaluation dictionary
            similarity_threshold: Minimum similarity score to consider accurate (default: 0.7)
            
        Returns:
            Dictionary with verification results including:
            - overall_accuracy: Average similarity score
            - per_agent_similarities: Similarity scores for each agent
            - accuracy_flag: Boolean indicating if all similarities meet threshold
            - warnings: List of agents with low similarity scores
        """
        if not final_responses:
            return {
                "overall_accuracy": 0.0,
                "per_agent_similarities": {},
                "accuracy_flag": False,
                "warnings": ["No agent responses provided"],
                "error": "No responses to verify"
            }
        
        # Combine all actual agent responses into a single text
        actual_responses_text = "\n\n".join([
            f"{agent}: {response}" 
            for agent, response in final_responses.items()
        ])
        
        # Get the judge's reasoning/evaluation text
        judge_text = evaluation.get("reasoning", "") + " " + evaluation.get("raw_evaluation", "")
        
        # If judge_text is empty, try to reconstruct from agent_evaluations
        if not judge_text.strip():
            agent_evals = evaluation.get("agent_evaluations", {})
            judge_text = " ".join([
                f"{agent}: {eval_result}" 
                for agent, eval_result in agent_evals.items()
            ])
        
        try:
            # Get embeddings for actual responses and judge's evaluation
            actual_embedding = self._get_embeddings([actual_responses_text])[0]
            judge_embedding = self._get_embeddings([judge_text])[0]
            
            # Calculate overall similarity
            overall_similarity = self._cosine_similarity(actual_embedding, judge_embedding)
            
            # Use more lenient thresholds since we're comparing evaluation text to response text
            # These are semantically different types, so perfect similarity isn't expected
            agent_threshold = similarity_threshold * 0.65  # 0.455 for default 0.7 threshold
            overall_threshold = similarity_threshold * 0.6  # 0.42 for default 0.7 threshold
            
            # Calculate per-agent similarities
            per_agent_similarities = {}
            warnings = []
            
            for agent, actual_response in final_responses.items():
                # Get embedding for this agent's actual response
                agent_actual_embedding = self._get_embeddings([actual_response])[0]
                
                # Extract what the judge said about this agent from the evaluation
                agent_eval = evaluation.get("agent_evaluations", {}).get(agent, "")
                agent_reasoning = ""
                
                # Improved extraction: get sentences mentioning this agent with more context
                if agent in judge_text:
                    # Split by sentences, but also look for agent mentions with more context
                    sentences = judge_text.replace('!', '.').replace('?', '.').split('.')
                    agent_sentences = []
                    for i, sentence in enumerate(sentences):
                        if agent.lower() in sentence.lower():
                            # Include surrounding context (previous and next sentence if available)
                            context_start = max(0, i - 1)
                            context_end = min(len(sentences), i + 2)
                            context = ". ".join(sentences[context_start:context_end])
                            agent_sentences.append(context)
                    
                    agent_reasoning = ". ".join(agent_sentences)
                
                # Combine agent evaluation and reasoning
                # Also include the actual response in the comparison to help with semantic matching
                judge_agent_text = f"{agent_eval}. {agent_reasoning}".strip()
                
                if judge_agent_text:
                    judge_agent_embedding = self._get_embeddings([judge_agent_text])[0]
                    similarity = self._cosine_similarity(agent_actual_embedding, judge_agent_embedding)
                else:
                    # If no judge text for this agent, use overall similarity
                    similarity = overall_similarity
                
                per_agent_similarities[agent] = similarity
                
                # Use the lenient threshold defined above
                if similarity < agent_threshold:
                    warnings.append(
                        f"Low similarity ({similarity:.2f}) for {agent}: "
                        f"judge's evaluation may not accurately reflect agent's response"
                    )
            
            accuracy_flag = all(
                sim >= agent_threshold 
                for sim in per_agent_similarities.values()
            ) and overall_similarity >= overall_threshold
            
            return {
                "overall_accuracy": float(overall_similarity),
                "per_agent_similarities": {k: float(v) for k, v in per_agent_similarities.items()},
                "accuracy_flag": accuracy_flag,
                "warnings": warnings,
                "similarity_threshold": similarity_threshold
            }
            
        except Exception as e:
            return {
                "overall_accuracy": 0.0,
                "per_agent_similarities": {},
                "accuracy_flag": False,
                "warnings": [f"Verification failed: {str(e)}"],
                "error": str(e)
            }
    
    def evaluate_conversation(self, claim: str, claim_type: str, 
                            final_responses: Dict[str, str],
                            conversation_history: List[Dict],
                            verify_accuracy: bool = True) -> Dict:
        """
        Evaluate a conversation and classify correctness.
        
        Args:
            claim: The original claim
            claim_type: Type of claim (ground_truth, false, debatable)
            final_responses: Final responses from each agent
            conversation_history: Full conversation history
            verify_accuracy: If True, verify judge accuracy using cosine similarity
        
        Returns:
            Dictionary with evaluation results, including accuracy verification if enabled
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

IMPORTANT: In your reasoning, for each agent, briefly summarize what they actually said in their final response. This helps verify that your evaluation accurately reflects their actual words.

Respond with a JSON object in the following format:
{{
    "overall_correctness": "correct|approximate|wrong",
    "agent_evaluations": {{
        "agent_name": "correct|approximate|wrong"
    }},
    "converged": true|false,
    "hallucinations_detected": true|false,
    "reasoning": "Detailed explanation that includes a brief summary of what each agent actually said in their final response. For example: 'The skeptic said [summary of their actual response]. The optimist argued [summary].' This helps ensure your evaluation accurately reflects their actual responses."
}}

Important: 
- For agent_evaluations, include an entry for each agent in the conversation.
- In your reasoning, include brief summaries of what each agent actually said to demonstrate you understood their responses."""

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
                
                # Verify accuracy if requested
                if verify_accuracy:
                    accuracy_verification = self.verify_judge_accuracy(
                        final_responses, 
                        evaluation
                    )
                    evaluation["accuracy_verification"] = accuracy_verification
                
                return evaluation
            except json.JSONDecodeError:
                # Fallback to old parsing if JSON is malformed
                evaluation = self._parse_evaluation(evaluation_text, final_responses)
                evaluation["raw_evaluation"] = evaluation_text
                
                # Verify accuracy if requested
                if verify_accuracy:
                    accuracy_verification = self.verify_judge_accuracy(
                        final_responses, 
                        evaluation
                    )
                    evaluation["accuracy_verification"] = accuracy_verification
                
                return evaluation
            
        except Exception as e:
            evaluation = {
                "overall_correctness": "unknown",
                "agent_evaluations": {},
                "converged": False,
                "hallucinations_detected": False,
                "error": str(e),
                "raw_evaluation": f"Error during evaluation: {str(e)}"
            }
            
            # Still try to verify if we have responses
            if verify_accuracy and final_responses:
                try:
                    accuracy_verification = self.verify_judge_accuracy(
                        final_responses, 
                        evaluation
                    )
                    evaluation["accuracy_verification"] = accuracy_verification
                except:
                    pass
            
            return evaluation
    
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

