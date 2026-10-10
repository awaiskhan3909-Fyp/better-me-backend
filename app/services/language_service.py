import re
import urllib.request
import urllib.parse
import json
import logging
from typing import NamedTuple

logger = logging.getLogger("better_me.language_service")

# High-frequency Roman Urdu words & phonetic markers
ROMAN_URDU_MARKERS = {
    "hai", "hain", "mein", "mai", "me", "mera", "meri", "mere", "mujhe", "mjy", "mjhy",
    "mje", "tum", "aap", "ap", "kya", "kyun", "kion", "kaise", "kese", "nahi", "nhi",
    "nh", "ni", "bohat", "bht", "buht", "bhot", "kuch", "kch", "raha", "rahi", "rahe",
    "karna", "krna", "hoga", "hogi", "hoge", "ho", "tha", "thi", "the", "bhi", "to",
    "toh", "par", "pe", "sirf", "zindagi", "zindgi", "dost", "parhai", "dil", "lag",
    "lagta", "lgta", "khud", "marne", "marna", "chahiye", "chahye", "shyd", "shyad",
    "shayad", "sab", "sb", "khatam", "khtm", "udas", "pareshan", "preshan", "gaya",
    "gya", "koi", "aur", "or", "hoon", "hn", "hun", "wala", "wali", "wale", "baat",
    "bt", "bura", "achha", "acha", "accha", "theek", "thik", "thk", "yaar", "yr",
    "naukri", "paisa", "paise", "rishta", "rishtey"
}

# Slang & Contraction Normalization Map
SLANG_MAP = [
    (re.compile(r"\bmjy\b", re.IGNORECASE), "mujhe"),
    (re.compile(r"\bmjhy\b", re.IGNORECASE), "mujhe"),
    (re.compile(r"\bmje\b", re.IGNORECASE), "mujhe"),
    (re.compile(r"\bmj\b", re.IGNORECASE), "mujhe"),
    (re.compile(r"\bnh\b", re.IGNORECASE), "nahi"),
    (re.compile(r"\bnhi\b", re.IGNORECASE), "nahi"),
    (re.compile(r"\bni\b", re.IGNORECASE), "nahi"),
    (re.compile(r"\bbht\b", re.IGNORECASE), "bohat"),
    (re.compile(r"\bbuht\b", re.IGNORECASE), "bohat"),
    (re.compile(r"\bbhot\b", re.IGNORECASE), "bohat"),
    (re.compile(r"\bkch\b", re.IGNORECASE), "kuch"),
    (re.compile(r"\blgta\b", re.IGNORECASE), "lagta"),
    (re.compile(r"\blgti\b", re.IGNORECASE), "lagti"),
    (re.compile(r"\blgte\b", re.IGNORECASE), "lagte"),
    (re.compile(r"\bkrna\b", re.IGNORECASE), "karna"),
    (re.compile(r"\bkrne\b", re.IGNORECASE), "karne"),
    (re.compile(r"\bkrta\b", re.IGNORECASE), "karta"),
    (re.compile(r"\bkrti\b", re.IGNORECASE), "karti"),
    (re.compile(r"\bkrte\b", re.IGNORECASE), "karte"),
    (re.compile(r"\bkr\b", re.IGNORECASE), "kar"),
    (re.compile(r"\bkro\b", re.IGNORECASE), "karo"),
    (re.compile(r"\bh\b", re.IGNORECASE), "hai"),
    (re.compile(r"\bhn\b", re.IGNORECASE), "hoon"),
    (re.compile(r"\bhun\b", re.IGNORECASE), "hoon"),
    (re.compile(r"\bsb\b", re.IGNORECASE), "sab"),
    (re.compile(r"\bkhtm\b", re.IGNORECASE), "khatam"),
    (re.compile(r"\bzindgi\b", re.IGNORECASE), "zindagi"),
    (re.compile(r"\bpreshan\b", re.IGNORECASE), "pareshan"),
    (re.compile(r"\bpreshani\b", re.IGNORECASE), "pareshani"),
    (re.compile(r"\btenshn\b", re.IGNORECASE), "tension"),
    (re.compile(r"\btens\b", re.IGNORECASE), "tension"),
    (re.compile(r"\bsmjh\b", re.IGNORECASE), "samajh"),
    (re.compile(r"\bgya\b", re.IGNORECASE), "gaya"),
    (re.compile(r"\bgyi\b", re.IGNORECASE), "gayi"),
    (re.compile(r"\bgye\b", re.IGNORECASE), "gaye"),
    (re.compile(r"\byr\b", re.IGNORECASE), "yaar"),
    (re.compile(r"\bplz\b", re.IGNORECASE), "please"),
    (re.compile(r"\bbt\b", re.IGNORECASE), "baat"),
    (re.compile(r"\bthk\b", re.IGNORECASE), "theek"),
    (re.compile(r"\bshyd\b", re.IGNORECASE), "shayad"),
    (re.compile(r"\bchahye\b", re.IGNORECASE), "chahiye"),
    # Replace dangerous crisis ambiguity to ensure clean translation
    (re.compile(r"\bkhud[\s\-]?kushi\b", re.IGNORECASE), "suicide"),
    (re.compile(r"\bkhudkushi\b", re.IGNORECASE), "suicide"),
]

# Direct Roman Urdu Crisis Lexicon (Zero-Tolerance Safety Gatekeeper)
ROMAN_URDU_CRISIS_PATTERNS = [
    re.compile(r"\bkhud[\s\-]?kushi\b", re.IGNORECASE),
    re.compile(r"\bjaan[\s\-]?de(na|ni|unga|ungi|u)\b", re.IGNORECASE),
    re.compile(r"\bmar[\s\-]?ja(na|ne|unga|ungi|u)\b", re.IGNORECASE),
    re.compile(r"\bmarna[\s\-]?h(ai)?\b", re.IGNORECASE),
    re.compile(r"\bmarna[\s\-]?chah(ta|ti|ye)\b", re.IGNORECASE),
    re.compile(r"\bmarne[\s\-]?ka[\s\-]?dil\b", re.IGNORECASE),
    re.compile(r"\bapni[\s\-]?jaan\b", re.IGNORECASE),
    re.compile(r"\bzeher\b", re.IGNORECASE),
    re.compile(r"\bzindagi[\s\-]?khatam\b", re.IGNORECASE),
    re.compile(r"\bkoi[\s\-]?faida[\s\-]?nahi[\s\-]?jeene\b", re.IGNORECASE),
    re.compile(r"\bsuicide\b", re.IGNORECASE),
]


class LanguageProcessResult(NamedTuple):
    original_text: str
    is_roman_urdu: bool
    normalized_text: str
    english_text: str
    is_vernacular_crisis: bool


class LanguageService:
    """
    Language Intelligence Service for Roman Urdu & Vernacular Normalization:
    1. Detects Roman Urdu and code-mixed inputs.
    2. Normalizes phonetic slang & abbreviations (e.g. 'mjy' -> 'mujhe', 'bht' -> 'bohat').
    3. Direct Vernacular Crisis Interceptor (Zero-bypass safety gatekeeper).
    4. Sub-200ms Semantic Translation to English for downstream BERT models.
    """

    def detect_roman_urdu(self, text: str) -> bool:
        """Detects whether text is Roman Urdu / Urdu Latin script."""
        if not text:
            return False

        # Check for actual Perso-Arabic Urdu script
        if re.search(r"[\u0600-\u06FF]", text):
            return True

        tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not tokens:
            return False

        match_count = sum(1 for t in tokens if t in ROMAN_URDU_MARKERS)
        ratio = match_count / max(len(tokens), 1)

        # Matched if 2+ markers, or 1 marker in a very short message (<=3 words), or >=20% tokens
        return match_count >= 2 or (len(tokens) <= 3 and match_count >= 1) or ratio >= 0.20

    def normalize_slang(self, text: str) -> str:
        """Normalizes informal phonetic contractions and abbreviations."""
        cleaned = text
        for pattern, replacement in SLANG_MAP:
            cleaned = pattern.sub(replacement, cleaned)
        return cleaned

    def check_vernacular_crisis(self, text: str) -> bool:
        """Direct regex check for unambiguous suicide / lethal self-harm in Roman Urdu."""
        normalized = text.lower()
        for pattern in ROMAN_URDU_CRISIS_PATTERNS:
            if pattern.search(normalized):
                return True
        return False

    def translate_to_english(self, text: str) -> str:
        """
        Translates Roman Urdu to clean English using fast Google Translate web endpoint.
        Times out after 2.5s with fallback to normalized text.
        """
        try:
            url = (
                "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q="
                + urllib.parse.quote(text)
            )
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                translated_segments = [seg[0] for seg in data[0] if seg and seg[0]]
                translated = " ".join(translated_segments).strip()
                if translated:
                    return translated
        except Exception as e:
            logger.warning(f"Translation failed or timed out: {e}. Falling back to normalized input.")

        return text

    def process_input(self, text: str) -> LanguageProcessResult:
        """Full language intelligence pipeline returning structured result."""
        raw_text = text.strip()
        is_ru = self.detect_roman_urdu(raw_text)
        is_crisis = self.check_vernacular_crisis(raw_text)

        if not is_ru:
            return LanguageProcessResult(
                original_text=raw_text,
                is_roman_urdu=False,
                normalized_text=raw_text,
                english_text=raw_text,
                is_vernacular_crisis=is_crisis
            )

        # 1. Normalize Slang
        normalized = self.normalize_slang(raw_text)

        # 2. Check crisis again on normalized form
        is_crisis = is_crisis or self.check_vernacular_crisis(normalized)

        # 3. Translate to English for downstream BERT models
        english = self.translate_to_english(normalized)

        return LanguageProcessResult(
            original_text=raw_text,
            is_roman_urdu=True,
            normalized_text=normalized,
            english_text=english,
            is_vernacular_crisis=is_crisis
        )


language_service = LanguageService()
