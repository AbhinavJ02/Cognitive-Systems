"""
Manages multi-agent conversations and tracks stance changes.
"""
from typing import List, Dict
from data.dataset import Claim
from openai import OpenAI
import config
import time
from datetime import datetime

class ConversationManager:
    def __init__(self, personalities: List[str] = None, model: str = None, 
                 temperature: float = None, max_turns: int = None, 
                 context_size: int = None):
        """
        Initialize conversation manager.
        
        Args:
            personalities: List of agent personalities
            model: OpenAI model to use
            temperature: Model temperature
            max_turns: Maximum number of conversation turns
            context_size: Maximum context size in tokens
        """
        self.personalities = personalities or list(config.AGENT_PERSONALITIES.keys())
        self.model = model or config.DEFAULT_MODEL
        self.temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        self.max_turns = max_turns or config.DEFAULT_MAX_TURNS
        self.context_size = context_size or config.DEFAULT_CONTEXT_SIZE
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Conversation state
        self.conversation_history: List[Dict] = []
        self.stance_tracking: Dict[str, List[Dict]] = {personality: [] for personality in self.personalities}
        self.turn_stances: List[Dict] = []
        self.current_turn = 0
        self.start_time = None
        self.end_time = None
    
    def _get_agent_prompt(self, personality: str, claim: str, conversation_so_far: List[Dict]) -> str:
        """Generate a prompt for an agent based on their personality and conversation context."""
        personality_config = config.AGENT_PERSONALITIES[personality]
        
        context = ""
        if conversation_so_far:
            context = "\n\nPrevious conversation:\n"
            for msg in conversation_so_far[-5:]:  # Last 5 messages for context
                context += f"{msg['agent']}: {msg['message']}\n"
        
        prompt = f"""You are a {personality_config['role']} agent in a debate.

Your personality traits: {', '.join(personality_config['personality_traits'])}

Your goal: {personality_config['goal']}

{personality_config['backstory']}

The claim being discussed is: "{claim}"

{context}

Provide your response to this claim. Be true to your personality. If this is your first response, give your initial stance. If others have spoken, respond to their points while maintaining your personality.

Your response:"""
        
        return prompt
    
    def _call_agent(self, personality: str, claim: str, conversation_so_far: List[Dict]) -> str:
        """Call an agent and get their response."""
        prompt = self._get_agent_prompt(personality, claim, conversation_so_far)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": config.AGENT_PERSONALITIES[personality]["backstory"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=min(500, self.context_size // 4)  # Reserve tokens for context
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error: {str(e)}]"
    
    def _determine_stance(self, message: str) -> str:
        """Determine stance label relative to the claim."""
        message_lower = message.lower()
        positive_keywords = [
            "agree", "support", "correct", "true", "believe", "affirm",
            "concur", "valid", "accurate", "accept", "yes"
        ]
        negative_keywords = [
            "disagree", "oppose", "incorrect", "false", "doubt",
            "reject", "refute", "invalid", "inaccurate", "no"
        ]
        hedging_keywords = [
            "maybe", "possibly", "uncertain", "unsure", "could be",
            "might", "depends", "not sure", "unclear", "ambiguous"
        ]

        if any(word in message_lower for word in positive_keywords):
            return "support"
        if any(word in message_lower for word in negative_keywords):
            return "oppose"
        if any(word in message_lower for word in hedging_keywords):
            return "uncertain"
        return "neutral"

    def _extract_stance(self, message: str, personality: str, claim_text: str = None) -> Dict:
        """Extract stance information from an agent's message using LLM."""
        stance = {
            "turn": self.current_turn,
            "round": (self.current_turn - 1) // len(self.personalities) + 1,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "sentiment": "neutral",
            "stance": "neutral",
            "personality": personality,
            "confidence": 0.0
        }
        
        # Use LLM for more accurate stance detection
        try:
            prompt = f"""Analyze the following agent message and determine the stance and sentiment regarding the claim.

Claim: "{claim_text if claim_text else 'N/A'}"

Agent Message:
{message}

Determine:
1. Stance: One of "support", "oppose", "uncertain", or "neutral" - indicates the agent's position on the claim
2. Sentiment: One of "positive", "negative", or "neutral" - indicates the emotional tone
3. Confidence: A number between 0.0 and 1.0 indicating how confident you are in the stance classification

Respond in JSON format:
{{
    "stance": "support|oppose|uncertain|neutral",
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0-1.0
}}"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing agent stances and sentiments in debates. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent stance classification
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            stance["stance"] = result.get("stance", "neutral").lower()
            stance["sentiment"] = result.get("sentiment", "neutral").lower()
            stance["confidence"] = float(result.get("confidence", 0.5))
            
        except Exception as e:
            # Fallback to keyword-based detection if LLM fails
            message_lower = message.lower()
            if any(word in message_lower for word in ["agree", "correct", "true", "accept", "believe"]):
                stance["sentiment"] = "positive"
            elif any(word in message_lower for word in ["disagree", "wrong", "false", "reject", "doubt"]):
                stance["sentiment"] = "negative"
            stance["stance"] = self._determine_stance(message)
            stance["confidence"] = 0.3  # Low confidence for fallback
        
        return stance

    def _record_turn_stance(self, stance: Dict):
        """Record stance information for turn-by-turn tracking."""
        self.turn_stances.append({
            "turn": stance["turn"],
            "round": stance["round"],
            "agent": stance["personality"],
            "stance": stance["stance"],
            "sentiment": stance["sentiment"],
            "confidence": stance.get("confidence", 0.0),
            "message": stance["message"],
            "timestamp": stance["timestamp"]
        })

    def _build_stance_summary(self) -> Dict:
        """Build aggregated stance summary."""
        summary = {
            "per_agent": {},
            "turns": self.turn_stances
        }
        for personality, stances in self.stance_tracking.items():
            summary["per_agent"][personality] = {
                "initial": stances[0]["stance"] if stances else "unknown",
                "final": stances[-1]["stance"] if stances else "unknown",
                "positive_turns": sum(1 for s in stances if s["sentiment"] == "positive"),
                "negative_turns": sum(1 for s in stances if s["sentiment"] == "negative"),
                "neutral_turns": sum(1 for s in stances if s["sentiment"] == "neutral"),
                "stance_history": stances
            }
        return summary
    
    def run_conversation(self, claim: Claim) -> Dict:
        """
        Run a conversation about a claim.
        
        Args:
            claim: The claim to discuss
        
        Returns:
            Dictionary with conversation results
        """
        self.start_time = time.time()
        self.conversation_history = []
        self.stance_tracking = {personality: [] for personality in self.personalities}
        self.turn_stances = []
        self.current_turn = 0
        
        claim_text = claim.text
        
        print(f"\n{'='*60}")
        print(f"Starting conversation about claim: {claim_text}")
        print(f"Claim type: {claim.claim_type.value}")
        print(f"Agents: {', '.join(self.personalities)}")
        print(f"{'='*60}\n")
        
        # Initial round: each agent gives their initial stance
        for personality in self.personalities:
            self.current_turn += 1
            response = self._call_agent(personality, claim_text, self.conversation_history)
            
            self.conversation_history.append({
                "turn": self.current_turn,
                "agent": personality,
                "message": response,
                "timestamp": datetime.now().isoformat()
            })
            
            stance = self._extract_stance(response, personality, claim_text)
            self.stance_tracking[personality].append(stance)
            self._record_turn_stance(stance)
            
            # Display stance and sentiment information
            stance_label = stance["stance"].upper()
            sentiment_label = stance["sentiment"].upper()
            confidence = stance.get("confidence", 0.0)
            print(f"[{personality.upper()}] (Turn {self.current_turn}, Round {stance['round']}): {response}")
            print(f"   └─ Stance: {stance_label} | Sentiment: {sentiment_label} | Confidence: {confidence:.2f}\n")
            
            if self.current_turn >= self.max_turns:
                break
        
        # Subsequent rounds: agents respond to each other
        while self.current_turn < self.max_turns:
            # Rotate through agents
            for personality in self.personalities:
                if self.current_turn >= self.max_turns:
                    break
                
                self.current_turn += 1
                response = self._call_agent(personality, claim_text, self.conversation_history)
                
                self.conversation_history.append({
                    "turn": self.current_turn,
                    "agent": personality,
                    "message": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                stance = self._extract_stance(response, personality, claim_text)
                self.stance_tracking[personality].append(stance)
                self._record_turn_stance(stance)
                
                # Display stance and sentiment information
                stance_label = stance["stance"].upper()
                sentiment_label = stance["sentiment"].upper()
                confidence = stance.get("confidence", 0.0)
                print(f"[{personality.upper()}] (Turn {self.current_turn}, Round {stance['round']}): {response}")
                print(f"   └─ Stance: {stance_label} | Sentiment: {sentiment_label} | Confidence: {confidence:.2f}\n")
        
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        # Compile final responses
        final_responses = {}
        for personality in self.personalities:
            if self.stance_tracking[personality]:
                final_responses[personality] = self.stance_tracking[personality][-1]["message"]
        
        result = {
            "claim": claim_text,
            "claim_type": claim.claim_type.value,
            "personalities": self.personalities,
            "conversation_history": self.conversation_history,
            "stance_tracking": self.stance_tracking,
            "stance_summary": self._build_stance_summary(),
            "final_responses": final_responses,
            "num_turns": self.current_turn,
            "duration_seconds": duration,
            "model": self.model,
            "temperature": self.temperature,
            "max_turns": self.max_turns,
            "context_size": self.context_size
        }
        
        return result

