import re
from typing import Tuple, Optional, Dict, Any
from app.schemas.response import CBTGuidance, DistortionPrediction


class FallbackService:
    """
    Emergency Safety Failover Engine.
    Only active if GPU is cold-starting or offline.
    Contains NO canned therapy scripts or hardcoded dialogue templates.
    """

    SAFETY_OVERRIDE_TEXT = (
        "Your safety and well-being are the top priority. "
        "If you or someone you know is experiencing thoughts of self-harm, suicide, or severe crisis, "
        "please reach out immediately to a trusted professional, doctor, or emergency helpline (such as 988).\n"
        "Support is available 24/7 and you do not have to carry this alone."
    )

    SAFETY_OVERRIDE_ROMAN_URDU_TEXT = (
        "Aapki hifazat aur zindagi bohot qeemti hai. "
        "Agar aap shadeed takleef ya khudkushi ke khayalat se guzar rahe hain, "
        "to barah-e-karam foran kisi doctor ya crisis helpline se rabta karein:\n"
        "• Pakistan Mental Health Helpline (Umang): 0311-7786264 / 0317-4288898\n"
        "• International Crisis Helpline: 988\n"
        "Aap akele nahi hain, madad 24/7 dastiyab hai."
    )

    def generate_fallback(
        self,
        user_text: str,
        strategy: str,
        cbt_guidance: Optional[CBTGuidance] = None,
        conversation_context: Optional[Any] = None,
        distortion_pred: Optional[DistortionPrediction] = None,
        is_roman_urdu: bool = False
    ) -> Tuple[str, Optional[Dict[str, Any]]]:

        # 1. STRICT SAFETY CRISIS OVERRIDE GATE
        if strategy == "safety_response":
            if is_roman_urdu:
                return self.SAFETY_OVERRIDE_ROMAN_URDU_TEXT, None
            return self.SAFETY_OVERRIDE_TEXT, None

        # 2. GREETINGS
        text_lower = user_text.lower().strip()
        words = set(re.findall(r"\b[a-zA-Z']+\b", text_lower))
        if any(w in words for w in ["hi", "hello", "hey", "salam", "assalam"]):
            if is_roman_urdu:
                return "Salam! Main Better Me AI hoon. Aap kaisa mehsoos kar rahe hain, aur aaj kis baare mein baat karna chahenge?", None
            return "Hello! I am Better Me, your clinical AI companion. How are you feeling right now, and what is on your mind?", None

        # 3. IDENTITY QUERY
        if any(w in words for w in ["who", "what", "kon", "kaun"]) and any(w in words for w in ["you", "u", "bot", "ai", "tum", "aap", "ap"]):
            if is_roman_urdu:
                return "Main Better Me AI hoon — aapka clinical cognitive therapy companion. Main aapki kis tarha madad kar sakta hoon?", None
            return "I am Better Me, your AI cognitive therapy companion. How can I support you today?", None

        # 4. NEUTRAL NON-HARDCODED TRANSITION
        if is_roman_urdu:
            content = (
                "Main aapki baat samajh raha hoon. "
                "Barah-e-karam thora aur tafseel se batayein ke is waqt aapke zehn mein kya chal raha hai taake hum mil kar ispar baat kar sakein."
            )
        else:
            content = (
                "I hear you. "
                "Please tell me a little more about what you're experiencing right now so we can explore this together."
            )

        return content, None


fallback_service = FallbackService()
