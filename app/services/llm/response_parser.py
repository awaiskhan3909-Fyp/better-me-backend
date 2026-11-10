import re
from typing import Tuple, Optional


class ResponseParser:
    """
    Validates, sanitizes, and formats LLM output for clinical dialogue.
    - Strips internal scaffolding, prompt markers, and leakage.
    - Strips stage tags (e.g. '(Socratic questioning)').
    - Enforces the Single Question Rule (at most ONE question per turn).
    - Deduplicates identical sentences.
    - Ensures proper sentence closure and quote stripping.
    """

    # Patterns that mark the beginning of leakage or prompt scaffolding to cut off
    TRUNCATION_MARKERS = [
        re.compile(r"\n*###.*$", re.DOTALL | re.IGNORECASE),
        re.compile(r"\n*\s*(?:\(|\[)?Note(?:\s*:|\s+to|\s+that|\s+for)[\s\S]*$", re.IGNORECASE),
        re.compile(r"\n*\s*By acknowledging the (?:patient|user|client)[\s\S]*$", re.IGNORECASE),
        re.compile(r"\n*\s*(?:The|This)\s+Socratic\s+questioning(?:\s+technique)?[\s\S]*$", re.IGNORECASE),
        re.compile(r"\n*\s*Please\s+continue\s+with[\s\S]*$", re.IGNORECASE),
        re.compile(r"(?:^|[.!?\n]\s*)Please\s+provide\s+(?:your|a)\s+[\s\S]*$", re.IGNORECASE),
        re.compile(r"(?:^|[.!?\n]\s*)Remember\s+to\s+maintain[\s\S]*$", re.IGNORECASE),
        re.compile(r"(?:^|[.!?\n]\s*)Ensure\s+your\s+answer\s+remains[\s\S]*$", re.IGNORECASE),
        re.compile(r"(?:^|[.!?\n]\s*)Within\s+the\s+boundaries\s+specified[\s\S]*$", re.IGNORECASE),
        re.compile(r"(?:^|[.!?\n]\s*)Thank\s+you\s+for\s+helping\s+patients[\s\S]*$", re.IGNORECASE),
        re.compile(r"(?:^|[.!?\n]\s*)(?:Your\s+response|Response)\s+should\s+not\s+exceed[\s\S]*$", re.IGNORECASE),
        re.compile(r"(?:^|[.!?\n]\s*)As\s+per\s+the\s+guidelines[\s\S]*$", re.IGNORECASE),
        re.compile(r"(?:^|[.!?\n]\s*)include\s+exactly\s+one\s+Socratic\s+question[\s\S]*$", re.IGNORECASE),
        re.compile(r"(?:^|[.!?\n]\s*)Guidelines:?[\s\S]*$", re.IGNORECASE),
        re.compile(r"(?:^|[.!?\n]\s*)Directives?:?[\s\S]*$", re.IGNORECASE),
        re.compile(r"\n*---+[\s\S]*$", re.MULTILINE),
        re.compile(r"\n*===+[\s\S]*$", re.MULTILINE),
    ]

    # Stage directions in standalone parentheses: (Socratic questioning...), (Validation...), etc.
    STAGE_DIRECTION_PATTERN = re.compile(
        r"\n+\s*\((?:Socratic|Validation|Empathetic|Cognitive|Balanced|Technique|Step|Stage|Therapist|Patient|Goal|Objective|Rationale|Strategy)[^)]*\)",
        re.IGNORECASE
    )

    PROMPT_LEAKAGE_HEADERS = [
        re.compile(r"^(?:system\s+instruction|directive|role|internal\s+context|authoritative\s+strategy|session\s+state)[:\s-].*?$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^###\s*Response:\s*", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^Better\s*Me:\s*", re.IGNORECASE),
        re.compile(r"^Therapist:\s*", re.IGNORECASE),
        re.compile(r"^Assistant:\s*", re.IGNORECASE),
    ]

    def validate_and_sanitize(self, raw_text: str, strategy: str = "cbt_support") -> Tuple[bool, str, Optional[str]]:
        if not raw_text or not raw_text.strip():
            return False, "", "empty_llm_response"

        clean_text = raw_text.strip()

        # 1. Clean leakage headers at the beginning
        for pat in self.PROMPT_LEAKAGE_HEADERS:
            clean_text = pat.sub("", clean_text).strip()

        # 2. Strip stage directions like (Socratic questioning...), (Validation...), etc.
        clean_text = self.STAGE_DIRECTION_PATTERN.sub(" ", clean_text).strip()

        # 3. Truncate at scaffolding or meta-commentary markers
        for marker in self.TRUNCATION_MARKERS:
            clean_text = marker.sub("", clean_text).strip()

        # 4. Strip any remaining inline stage tags like (Empathetic validation) or (Socratic questioning)
        clean_text = re.sub(
            r"\s*\((?:Socratic questioning|Empathetic validation|Validation|Cognitive reframing|Balanced reframe|Stage\s+\d+|Step\s+\d+)[^)]*\)\s*",
            " ",
            clean_text,
            flags=re.IGNORECASE
        ).strip()

        # 5. Remove surrounding and dangling quotes
        clean_text = clean_text.strip()
        if (clean_text.startswith('"') and clean_text.endswith('"')) or (clean_text.startswith("'") and clean_text.endswith("'")):
            clean_text = clean_text[1:-1].strip()
        if clean_text.startswith(('"', "'")):
            clean_text = clean_text[1:].strip()
        if clean_text.endswith(('"', "'")):
            clean_text = clean_text[:-1].strip()

        # 6. Deduplicate repeated sentences
        clean_text = self._deduplicate_sentences(clean_text)

        # 7. Enforce the Single Question Rule (at most ONE question per turn)
        clean_text = self._enforce_single_question(clean_text)

        # 8. Ensure the response ends on a complete sentence if truncated mid-sentence
        clean_text = self._ensure_complete_sentence(clean_text)

        words = clean_text.split()
        if len(words) < 2:
            return False, clean_text, "response_too_short"

        return True, clean_text, None

    def _enforce_single_question(self, text: str) -> str:
        """
        Ensures the assistant asks at most ONE question per turn.
        If multiple question marks exist, keeps the final focused question
        and converts earlier questions to declarative sentences.
        """
        if text.count("?") <= 1:
            return text

        sentences = re.split(r'(?<=[.?!])\s+', text)
        question_indices = [i for i, s in enumerate(sentences) if "?" in s]
        if len(question_indices) <= 1:
            return text

        last_q_idx = question_indices[-1]
        result_sentences = []
        for i, s in enumerate(sentences):
            if i in question_indices and i != last_q_idx:
                # Convert earlier question into a declarative statement by replacing ? with .
                softened = s.replace("?", ".").strip()
                result_sentences.append(softened)
            else:
                result_sentences.append(s)

        return " ".join(result_sentences).strip()

    def _deduplicate_sentences(self, text: str) -> str:
        """Removes consecutive identical sentences generated by degenerate LLM loops."""
        if not text:
            return text
        sentences = re.split(r'(?<=[.?!])\s+', text)
        deduped = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            if not deduped or s_clean.lower() != deduped[-1].strip().lower():
                deduped.append(s_clean)
        return " ".join(deduped).strip()

    def _ensure_complete_sentence(self, text: str) -> str:
        """
        If text was cut off mid-thought without punctuation, trim back to the last complete sentence.
        """
        if not text:
            return text

        # If it already ends with valid punctuation, return
        if text[-1] in [".", "!", "?", "—", '"', "'"]:
            return text

        # Find the last terminal punctuation
        match = re.search(r"^(.*[.!?])\s*[^.!?]*$", text, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            # If the candidate has substantial length, return it
            if len(candidate.split()) >= 4:
                return candidate

        # Otherwise add period
        return text + "."

    def extract_question(self, text: str) -> Optional[str]:
        """Extracts the single asked question from the assistant response if present."""
        if not text or "?" not in text:
            return None
        sentences = re.split(r'(?<=[.?!])\s+', text)
        for s in reversed(sentences):
            if "?" in s:
                return s.strip()
        return None


response_parser = ResponseParser()
