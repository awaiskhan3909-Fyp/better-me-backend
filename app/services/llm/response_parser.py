import re
from typing import Tuple, Optional


class ResponseParser:
    """
    Validates and sanitizes raw LLM output text.
    Safely cleans prompts and ensures natural conversational delivery.
    """

    PROMPT_LEAKAGE_PATTERNS = [
        re.compile(r"^(?:system\s+instruction|directive|role|internal\s+context|authoritative\s+strategy)[:\s-].*?$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"###\s*Response:\s*", re.IGNORECASE)
    ]

    def validate_and_sanitize(self, raw_text: str, strategy: str) -> Tuple[bool, str, Optional[str]]:
        if not raw_text or not raw_text.strip():
            return False, "", "empty_llm_response"

        clean_text = raw_text.strip()

        # Clean leakage headers if present
        for pat in self.PROMPT_LEAKAGE_PATTERNS:
            clean_text = pat.sub("", clean_text).strip()

        # Remove surrounding quotes if model wrapped response in quotes
        if (clean_text.startswith('"') and clean_text.endswith('"')) or (clean_text.startswith("'") and clean_text.endswith("'")):
            clean_text = clean_text[1:-1].strip()

        words = clean_text.split()
        if len(words) < 2:
            return False, clean_text, "response_too_short"

        return True, clean_text, None


response_parser = ResponseParser()
