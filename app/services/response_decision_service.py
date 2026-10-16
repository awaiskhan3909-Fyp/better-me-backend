import re
from enum import Enum
from typing import List, Optional, Set
from pydantic import BaseModel

from app.schemas.response import SafetyPrediction, DistortionPrediction, EntityItem
from app.services.conversation_manager import ConversationContext


class ResponseStrategy(str, Enum):
    SAFETY_RESPONSE = "safety_response"
    CBT_SUPPORT = "cbt_support"
    EXPLORATORY_FOLLOW_UP = "exploratory_follow_up"
    CLARIFICATION = "clarification"
    NORMAL_CONVERSATION = "normal_conversation"


class DecisionPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    NORMAL = "normal"


class DecisionContext(BaseModel):
    user_text: str
    safety_result: SafetyPrediction
    distortion_result: DistortionPrediction
    entities: List[EntityItem]
    conversation_context: Optional[ConversationContext] = None


class ResponseDecision(BaseModel):
    strategy: ResponseStrategy
    priority: DecisionPriority
    reason: str
    distortion_detected: Optional[str] = None
    safety_risk_level: str = "Safe"
    decision_version: str = "v2"


class ResponseDecisionService:
    """
    Response Decision Engine with Context & Evidence Gate:
    1. Safety model output (Priority 1 - Critical)
    2. Greetings & Help-Seeking requests (Priority 2 - Normal Chat)
    3. Context/Evidence Gate for CBT Support (Priority 3 - Requires explicit distorted thought evidence)
    4. Ambiguous / Short Input (Priority 4 - Clarification)
    5. Event + Emotion / Personal Sharing (Priority 5 - Exploratory Follow-Up)
    
    Does NOT treat ML distortion classification as automatic CBT authorization.
    """

    GREETINGS = {
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
        "good evening", "howdy", "sup", "yo", "how are you", "hows it going",
        "thanks", "thank you", "bye", "goodbye"
    }

    HELP_REQUESTS = {
        "could you help me", "can you help me", "help me", "i need help",
        "can u help me", "could u help me", "please help me", "will you help me",
        "can you assist me", "could you assist me"
    }

    NON_DISTORTIONS = {"none", "normal", "no distortion", "general", "no_distortion", "null"}

    CBT_EVIDENCE_WORDS = {
        "never", "always", "every", "everyone", "nobody", "nothing", "everything",
        "completely", "totally", "entirely", "impossible", "useless", "worthless",
        "ruined", "ruins", "failure", "unfit", "hopeless", "pointless", "disaster",
        "tabah", "barbaad", "khatam", "bekaar", "mayoos", "failed at life", "ruined forever"
    }

    DISTORTED_LINK_PATTERNS = [
        r"\bnot\s+fit\s+(?:in|for)\s+this\s+world\b",
        r"\bdont\s+belong\s+(?:in|to)\s+this\s+world\b",
        r"\b(?:my\s+life|future)\s+(?:is\s+)?(?:ruined|ruins|over|destroyed)\b",
        r"\bi\s+think\s+my\s+life\s+(?:ruins|is\s+ruined)\b",
        r"\b(?:failed|fail)\s+(?:at|in)\s+life\b",
        r"\bno\s+hope\s+for\s+(?:me|future|my\s+life)\b",
        r"\b(?:dunya|zindagi)\s+(?:ke\s+qabil|me\s+nahi|tabah|barbaad)\b",
        r"\bso\s+(?:i\s+will|obviously|which\s+means|this\s+proves|my\s+life|our\s+friendship|i\s+know|everyone)\b",
        r"\b(?:know|sure)\s+(?:i\s+will\s+never|they\s+all|nobody\s+wants|everyone\s+hates)\b",
        r"\b(?:obviously|clearly)\s+(?:everyone|nobody|my\s+life|our\s+friendship|over)\b",
        r"\b(?:i\s+am|im)\s+a\s+(?:failure|disaster|loser|useless\s+person)\b",
        r"\b(?:ruined|over)\s+forever\b"
    ]

    def has_cbt_evidence(self, text_lower: str, words: List[str]) -> bool:
        """
        Evaluates whether the user text contains explicit cognitive distortion structure,
        such as absolute statements, distorted causal claims, existential defectiveness, or catastrophizing links.
        """
        if any(marker in words for marker in self.CBT_EVIDENCE_WORDS):
            return True

        for pattern in self.DISTORTED_LINK_PATTERNS:
            if re.search(pattern, text_lower):
                return True

        return False

    def evaluate_decision(self, context: DecisionContext) -> ResponseDecision:
        user_text = context.user_text.strip()
        text_lower = user_text.lower()
        text_clean = re.sub(r'[^\w\s]', '', text_lower).strip()
        words = re.findall(r'\w+', text_clean)

        # -------------------------------------------------------------
        # RULE 1 — SAFETY FIRST (Highest Priority - Critical)
        # -------------------------------------------------------------
        safety_risk = context.safety_result.risk_level.lower()
        
        # Comprehensive Crisis & Self-Harm Patterns (handles common typos like sucide/suicde, and 1st/3rd person)
        CRISIS_PATTERNS = [
            r"\b(?:suicid|sucid|suicd)\w*\b",  # suicide, suicidal, sucide, sucidal, suicde
            r"\b(?:kill|hurt|harm|cut|poison|hang|shoot|end)\s+(?:my|his|her|their|one'?s)?\s*(?:self|life)\b",
            r"\b(?:want|wants|wishing)\s+to\s+die\b",
            r"\b(?:better\s+off\s+dead|tired\s+of\s+living|no\s+reason\s+to\s+live)\b",
            r"\bend(?:ing)?\s+it\s+all\b",
            r"\bself[\s\-]harm\w*\b"
        ]
        is_crisis_text = any(bool(re.search(pat, text_lower)) for pat in CRISIS_PATTERNS)

        if safety_risk in ["high risk", "high"] or context.safety_result.needs_safety_alert or is_crisis_text:
            return ResponseDecision(
                strategy=ResponseStrategy.SAFETY_RESPONSE,
                priority=DecisionPriority.CRITICAL,
                reason="high_risk_safety_alert",
                safety_risk_level="High Risk" if is_crisis_text else context.safety_result.risk_level,
                distortion_detected=context.distortion_result.predicted_class
            )

        # -------------------------------------------------------------
        # RULE 2A-1 — GRATITUDE, AFFIRMATION & SESSION WRAP-UP
        # (e.g. "thank you", "thanks", "ok thank you i love you", "shukriya", "jazakallah", "bye")
        # -------------------------------------------------------------
        GRATITUDE_WRAPUP_PATTERNS = [
            r"\b(?:thanks|thank\s+you|thx|tysm|shukriya|jazakallah|bohut\s+shukriya|shukran)\b",
            r"\b(?:love\s+you|appreciate\s+you|appreciate\s+it|you\s+helped\s+me|feeling\s+better\s+now|helped\s+a\s+lot)\b",
            r"\b(?:bye|goodbye|allah\s+hafiz|see\s+you|take\s+care|talk\s+later)\b"
        ]
        is_gratitude = any(bool(re.search(pat, text_lower)) for pat in GRATITUDE_WRAPUP_PATTERNS)

        has_prior_history = bool(
            context.conversation_context
            and context.conversation_context.has_history
            and context.conversation_context.history_count > 0
        )

        if is_gratitude:
            return ResponseDecision(
                strategy=ResponseStrategy.NORMAL_CONVERSATION,
                priority=DecisionPriority.NORMAL,
                reason="gratitude_or_session_wrap_up" if has_prior_history else "casual_conversation_or_greeting",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level
            )

        # -------------------------------------------------------------
        # RULE 2A-2 — GREETINGS & CASUAL CHAT INTROS
        # -------------------------------------------------------------
        GREETING_PATTERNS = [
            r"\b(?:hi|hello|hey|salam|assalam|assalamoalaikum|assalamualaykum|greetings|howdy|sup|yo)\b",
            r"\b(?:good\s+morning|good\s+afternoon|good\s+evening|good\s+night)\b",
            r"\b(?:how\s+are\s+you|hows\s+it\s+going|kese\s+ho|kaise\s+ho|kya\s+haal\s+hai|kya\s+hal)\b"
        ]
        is_greeting = (
            text_clean in self.GREETINGS
            or any(bool(re.search(pat, text_lower)) for pat in GREETING_PATTERNS)
            or any(text_clean.startswith(g) for g in ["hi ", "hello", "hey", "good morning", "good afternoon", "good evening", "how are you"])
        )

        if is_greeting:
            return ResponseDecision(
                strategy=ResponseStrategy.NORMAL_CONVERSATION,
                priority=DecisionPriority.NORMAL,
                reason="casual_conversation_or_greeting",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level
            )

        # -------------------------------------------------------------
        # RULE 2B — HELP-SEEKING REQUESTS
        # -------------------------------------------------------------
        is_help_request = (
            text_clean in self.HELP_REQUESTS
            or any(text_clean.startswith(h) for h in ["could you help", "can you help", "help me", "i need help", "can u help", "could u help"])
        )

        if is_help_request:
            return ResponseDecision(
                strategy=ResponseStrategy.NORMAL_CONVERSATION,
                priority=DecisionPriority.NORMAL,
                reason="help_seeking_request",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level
            )

        # -------------------------------------------------------------
        # RULE 2C — SOLUTION & ACTION-SEEKING REQUESTS IN THERAPY
        # -------------------------------------------------------------
        SOLUTION_REQUEST_PATTERNS = [
            r"\b(?:tell\s+me\s+(?:the\s+)?solution|what\s+is\s+the\s+solution|solution\s+of\s+this)\b",
            r"\b(?:what\s+(?:should|can)\s+i\s+do|how\s+(?:can|do)\s+i\s+(?:fix|solve|handle|deal\s+with)\s+this)\b",
            r"\b(?:give\s+me\s+(?:a\s+)?solution|how\s+to\s+(?:fix|solve|recover|overcome))\b",
            r"\b(?:iska\s+(?:koi\s+)?hal|kya\s+karun|kya\s+karein|hal\s+batao|solution\s+batao|kuch\s+mashwara|koi\s+rasta)\b"
        ]
        is_solution_request = any(bool(re.search(pat, text_lower)) for pat in SOLUTION_REQUEST_PATTERNS)

        if is_solution_request:
            return ResponseDecision(
                strategy=ResponseStrategy.CBT_SUPPORT,
                priority=DecisionPriority.HIGH,
                reason="solution_seeking_intervention",
                distortion_detected=context.distortion_result.predicted_class,
                safety_risk_level=context.safety_result.risk_level
            )

        # -------------------------------------------------------------
        # RULE 2D — IDENTITY, BOT INTRO & CAPABILITY INQUIRIES
        # e.g., "who are you", "who r u", "what are you", "kon ho ap bhai", "tum kon ho", "kya ho tum"
        # -------------------------------------------------------------
        IDENTITY_PATTERNS = [
            r"\b(?:who\s+(?:are|r)\s+(?:you|u)|who\s+is\s+this|what\s+are\s+you|what\s+is\s+your\s+name)\b",
            r"\b(?:kon\s+ho|kaun\s+ho|kya\s+ho\s+tum|kya\s+ho\s+aap|kya\s+ho\s+ap|ap\s+kon\s+ho|aap\s+kaun\s+ho|tum\s+kon\s+ho|tum\s+kaun\s+ho)\b",
            r"\b(?:kon\s+hain\s+aap|apka\s+naam\s+kya\s+hai|aapka\s+naam|naam\s+kya\s+hai)\b",
            r"\b(?:who\s+created\s+you|who\s+made\s+you|tell\s+me\s+about\s+yourself|apna\s+intro)\b"
        ]
        is_identity_query = any(bool(re.search(pat, text_lower)) for pat in IDENTITY_PATTERNS) or (
            any(w in words for w in ["who", "what", "kon", "kaun", "kya"])
            and any(w in words for w in ["you", "u", "bot", "ai", "system", "tum", "aap", "ap", "bhai", "bro"])
            and not any(w in words for w in ["sad", "depressed", "anxious", "failed", "die", "suicide", "exam", "problem", "marks"])
        )

        if is_identity_query:
            return ResponseDecision(
                strategy=ResponseStrategy.NORMAL_CONVERSATION,
                priority=DecisionPriority.NORMAL,
                reason="identity_and_role_query",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level
            )

        # -------------------------------------------------------------
        # RULE 3 — CONTEXT & EVIDENCE GATE FOR CBT SUPPORT
        # Signal (Classifier) + Evidence (Explicit Distorted Thought Pattern)
        # -------------------------------------------------------------
        predicted_distortion = context.distortion_result.predicted_class
        confidence = context.distortion_result.confidence

        is_signal_present = (
            predicted_distortion
            and predicted_distortion.lower() not in self.NON_DISTORTIONS
            and confidence >= 0.50
        )

        has_evidence = self.has_cbt_evidence(text_lower, words)

        # Exclude questions directed to the bot from being classified as distortions
        is_question_to_bot = bool(
            re.search(r"\b(?:who|what|why|how|when|where|kon|kaun|kya|kyun|kese|kaise)\b", text_lower)
            and any(w in words for w in ["you", "u", "bot", "ai", "tum", "aap", "ap", "bhai"])
        )

        has_distress = any(w in words for w in [
            "sad", "depressed", "anxiety", "anxious", "scared", "fear", "fail", "failed", "failure",
            "cry", "crying", "hate", "hates", "hopeless", "worthless", "useless", "alone", "upset",
            "preshan", "pareshan", "udas", "khof", "dar", "mushkil", "nakam", "ruined", "ruins", "dunya", "life"
        ])

        # Cognitive distortion triggers CBT support if evidence is present, or if distress is expressed, or with high confidence
        if not is_question_to_bot and is_signal_present and (has_evidence or has_distress or confidence >= 0.65):
            return ResponseDecision(
                strategy=ResponseStrategy.CBT_SUPPORT,
                priority=DecisionPriority.HIGH,
                reason="cognitive_distortion_with_evidence",
                distortion_detected=predicted_distortion,
                safety_risk_level=context.safety_result.risk_level
            )

        # -------------------------------------------------------------
        # RULE 4 — CLARIFICATION FOR AMBIGUOUS / SHORT INPUTS
        # -------------------------------------------------------------
        if len(words) <= 2:
            return ResponseDecision(
                strategy=ResponseStrategy.CLARIFICATION,
                priority=DecisionPriority.MEDIUM,
                reason="ambiguous_short_input",
                distortion_detected=predicted_distortion,
                safety_risk_level=context.safety_result.risk_level
            )

        # -------------------------------------------------------------
        # RULE 5 — EXPLORATORY FOLLOW-UP FOR EVENT + EMOTION / PERSONAL SHARING
        # -------------------------------------------------------------
        personal_pronouns = {"i", "me", "my", "myself", "we", "our", "feel", "feeling", "thought", "thinking", "good", "bad", "sad", "upset"}
        has_personal_sharing = any(w in personal_pronouns for w in words) or len(words) >= 3

        if has_personal_sharing:
            return ResponseDecision(
                strategy=ResponseStrategy.EXPLORATORY_FOLLOW_UP,
                priority=DecisionPriority.NORMAL,
                reason="personal_event_or_emotion_sharing",
                distortion_detected=predicted_distortion,
                safety_risk_level=context.safety_result.risk_level
            )

        # -------------------------------------------------------------
        # DEFAULT — NORMAL CONVERSATION
        # -------------------------------------------------------------
        return ResponseDecision(
            strategy=ResponseStrategy.NORMAL_CONVERSATION,
            priority=DecisionPriority.NORMAL,
            reason="casual_conversation",
            distortion_detected=predicted_distortion,
            safety_risk_level=context.safety_result.risk_level
        )


response_decision_service = ResponseDecisionService()
