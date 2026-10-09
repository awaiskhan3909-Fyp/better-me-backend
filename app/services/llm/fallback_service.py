import re
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

    ROTATING_EXPLORATORY_PROMPTS = [
        "I'm listening and here with you. What specific moment or thought has been feeling the most difficult today?",
        "Thank you for sharing that with me. Could you walk me through what's been happening around this situation?",
        "It takes courage to open up about these things. When you notice these feelings coming up, what thoughts usually accompany them?",
        "I want to make sure I understand what you're experiencing. How long have you been carrying this stress?",
        "I hear how much this is impacting you. If we were to unpack this one step at a time, where would you like to start?"
    ]

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

        text_lower = user_text.lower()
        words = set(re.findall(r"\b[a-zA-Z']+\b", text_lower))

        # 1. Identity & Role Queries
        if any(w in words for w in ["who", "what"]) and any(w in words for w in ["you", "bot", "ai", "system", "better"]):
            content = (
                "I am Better Me, your clinical AI cognitive therapy companion. "
                "I'm designed to help you explore your thoughts, identify unhelpful cognitive patterns, "
                "and work through evidence-based CBT exercises in a safe, confidential space. "
                "How can I support you right now?"
            )
            return content, None

        # 2. Greetings
        if any(w in words for w in ["hi", "hello", "hey", "assalam", "salam"]):
            content = "Hello! I'm here to listen and help you work through any thoughts or situations on your mind today. How are you feeling right now?"
            return content, None

        # 3. How are you
        if "how are you" in text_lower or "hows it going" in text_lower or "how are you doing" in text_lower:
            content = "I'm doing well, thank you for asking! I'm ready to focus entirely on you. What's on your mind today?"
            return content, None

        # 4. Acute Overwhelm / Crisis Slang
        if any(w in words for w in ["fucked", "screwed", "ruined", "dying", "doomed", "mess", "shit", "hell"]):
            content = (
                "I hear how intensely overwhelmed and stressed you're feeling right now. "
                "When everything feels like it's crashing down, it can be truly exhausting. "
                "Can you take a deep breath with me and tell me what the biggest trigger was today?"
            )
            return content, None

        # 5. Sleep, Insomnia & Physical Fatigue
        if any(w in words for w in ["sleep", "insomnia", "sleepless", "awake", "tired", "exhausted", "drained", "cycle"]):
            content = (
                "Disrupted sleep and physical exhaustion make emotional stress feel so much heavier. "
                "When your sleep cycle is affected, your mind doesn't get the rest it needs to process difficulties. "
                "What thoughts tend to race through your mind when you're trying to rest?"
            )
            return content, None

        # 6. Helplessness & Hopelessness
        if any(w in words for w in ["helpless", "hopeless", "useless", "worthless", "stuck", "pointless", "alone", "lonely"]):
            content = (
                "Feeling helpless is a really painful place to be, and it's completely understandable that you feel drained. "
                "Even when our thoughts tell us nothing can change, that is often distress talking rather than the whole reality. "
                "What feels like the heaviest part of this situation right now?"
            )
            return content, None

        # 7. Academic / Exam Stress
        if any(w in words for w in ["exam", "test", "study", "studies", "result", "grade", "college", "school", "university", "paper", "failed", "marks"]):
            content = "Having a tough time with an exam or studies can feel really discouraging and stressful. What thoughts have been running through your mind since this happened?"
            return content, None

        # 8. Interpersonal / Relationship Issues (Strict word boundaries)
        if any(w in words for w in ["friend", "father", "mother", "brother", "sister", "mom", "dad", "partner", "boyfriend", "girlfriend", "boss", "colleague"]):
            content = "It sounds like relationship dynamics or concerns about someone close to you are playing a big role. What has been happening between you two, and how is it affecting you?"
            return content, None

        # 9. Work & Career Stress
        if any(w in words for w in ["work", "job", "career", "office", "interview", "salary", "fired", "promotion"]):
            content = "Dealing with challenges at work or career can feel really demanding. Could you tell me what specific part of this situation is causing the most stress?"
            return content, None

        # 10. General Exploratory (Rotating to prevent repetition)
        prompt_idx = abs(hash(user_text)) % len(self.ROTATING_EXPLORATORY_PROMPTS)
        content = self.ROTATING_EXPLORATORY_PROMPTS[prompt_idx]
        return content, None


fallback_service = FallbackService()

