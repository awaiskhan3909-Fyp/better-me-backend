from typing import Tuple, Optional, Dict, Any
from app.schemas.response import CBTGuidance, DistortionPrediction


class FallbackService:
    """
    Deterministic Fallback Response Service used when LLM API is unavailable,
    times out, or generates invalid responses.
    """

    SAFETY_OVERRIDE_TEXT = (
        "Your safety and the well-being of those you care about are incredibly important. "
        "If you or someone you know is experiencing overwhelming distress, suicidal thoughts, or thoughts of self-harm, "
        "please reach out immediately to a trusted professional, doctor, or emergency crisis helpline (such as 988 or local emergency services). "
        "Support is available 24/7 and you do not have to carry this alone."
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
            text_lower = user_text.lower()
            if any(k in text_lower for k in ["exam", "test", "study", "studies", "result", "grade", "college", "school", "university", "paper", "failed", "marks"]):
                content = "Failing or having a tough time with an exam can feel really discouraging and stressful. What thoughts have been running through your mind since this happened?"
            elif any(k in text_lower for k in ["he ", "she ", "friend", "father", "mother", "brother", "sister", "someone", "they"]):
                content = "It sounds like you are really concerned about them. What has been happening in their situation, and how has this been affecting you?"
            elif any(k in text_lower for k in ["sad", "depress", "lonely", "alone", "crying", "hopeless", "unhappy", "hurts", "pain"]):
                content = "I'm really sorry things feel this heavy and difficult right now. What has been weighing on you the most today?"
            elif any(k in text_lower for k in ["work", "job", "boss", "career", "office", "interview"]):
                content = "Dealing with challenges at work or career can feel really demanding. Could you tell me what specific part of this situation is causing the most stress?"
            else:
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
