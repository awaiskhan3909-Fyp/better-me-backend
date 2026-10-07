import re
from typing import Tuple, Optional


class ResponseParser:
    """
    Validates and sanitizes raw LLM output text before sending it down the pipeline.
    """

    PROMPT_LEAKAGE_MARKERS = [
        "authoritative strategy",
        "system instruction",
        "cbt template grounding",
        "internal context",
        "detected thought pattern",
        "directive:",
        "role: you are",
        "safety status:"
    ]

    def validate_and_sanitize(self, raw_text: str, strategy: str) -> Tuple[bool, str, Optional[str]]:
        if not raw_text or not raw_text.strip():
            return False, "", "empty_llm_response"

        clean_text = raw_text.strip()

        # Check prompt leakage
        lower_text = clean_text.lower()
        for marker in self.PROMPT_LEAKAGE_MARKERS:
            if marker in lower_text:
                return False, clean_text, f"prompt_leakage_detected ({marker})"

        # Check response length bounds based on strategy
        words = clean_text.split()
        if len(words) < 2:
            return False, clean_text, "response_too_short"

        if strategy in ["normal_conversation", "clarification"] and len(words) > 80:
            # Truncate or flag if unreasonably verbose for simple turns
            pass

        # Remove surrounding quotes if model added them
        if (clean_text.startswith('"') and clean_text.endswith('"')) or (clean_text.startswith("'") and clean_text.endswith("'")):
            clean_text = clean_text[1:-1].strip()

        return True, clean_text, None


response_parser = ResponseParser()
