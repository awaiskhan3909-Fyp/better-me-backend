from typing import Tuple, Optional, Dict, Any
from app.schemas.response import CBTGuidance, DistortionPrediction


class FallbackService:
    """
    Deterministic Fallback Response Service used when LLM API is unavailable,
    times out, or generates invalid responses.
    """

    SAFETY_OVERRIDE_TEXT = (
        "Your safety and well-being are incredibly important. If you are experiencing "
        "overwhelming distress or thoughts of self-harm, please reach out immediately to a trusted "
        "professional or emergency crisis helpline (such as 988 or local emergency services)."
    )

    def generate_fallback(
        self,
        user_text: str,
        strategy: str,
        cbt_guidance: Optional[CBTGuidance] = None
    ) -> Tuple[str, Optional[Dict[str, Any]]]:

        if strategy == "safety_response":
            return self.SAFETY_OVERRIDE_TEXT, None

        elif strategy == "cbt_support" and cbt_guidance and cbt_guidance.balanced_thought_guidance:
            return cbt_guidance.balanced_thought_guidance, cbt_guidance.model_dump()

        elif strategy == "exploratory_follow_up":
            content = "Thank you for sharing what's on your mind. Could you tell me a bit more about what's been happening around this situation?"
            return content, None

        elif strategy == "clarification":
            content = "Could you share a bit more context about what you mean? I want to make sure I understand your situation properly."
            return content, None

        else:  # normal_conversation
            clean_lower = user_text.lower().strip("?.! ")
            if clean_lower in ["how are you", "how are you doing", "hows it going"]:
                content = "I'm doing well, thank you for asking! How are you feeling right now?"
            else:
                content = "Hello! I'm here to listen and help you work through any thoughts or situations on your mind today. How are you feeling right now?"
            return content, None


fallback_service = FallbackService()
