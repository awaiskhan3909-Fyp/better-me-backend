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

    ABSOLUTE_MARKERS = {
        "never", "always", "every", "everyone", "nobody", "nothing", "everything",
        "completely", "totally", "entirely", "impossible", "useless", "worthless",
        "ruined", "failed at life", "ruined forever"
    }

    DISTORTED_LINK_PATTERNS = [
        r"\bso\s+(i\s+will|obviously|which\s+means|this\s+proves|my\s+life|our\s+friendship|i\s+know|everyone)\b",
        r"\b(know|sure)\s+(i\s+will\s+never|they\s+all|nobody\s+wants|everyone\s+hates)\b",
        r"\b(obviously|clearly)\s+(everyone|nobody|my\s+life|our\s+friendship|over)\b",
        r"\b(i\s+am|im)\s+a\s+(failure|disaster|loser|useless\s+person)\b",
        r"\b(ruined|over)\s+forever\b"
    ]

    def has_cbt_evidence(self, text_lower: str, words: List[str]) -> bool:
        """
        Evaluates whether the user text contains explicit cognitive distortion structure,
        such as absolute statements, distorted causal claims, or catastrophizing links.
        Merely expressing an emotion or describing an event does NOT pass the gate.
        """
        # Check absolute markers
        if any(marker in words for marker in ["never", "always", "nobody", "everyone", "nothing", "everything", "useless", "worthless"]):
            return True

        # Check explicit distorted link regex patterns
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
        # RULE 2A — GREETINGS & CASUAL CHAT
        # -------------------------------------------------------------
        is_greeting = (
            text_clean in self.GREETINGS
            or any(text_clean.startswith(g) for g in ["hi ", "hello", "hey", "good morning", "good afternoon", "good evening", "how are you"])
        )

        if is_greeting:
            return ResponseDecision(
                strategy=ResponseStrategy.NORMAL_CONVERSATION,
                priority=DecisionPriority.NORMAL,
                reason="casual_conversation_or_greeting",
                distortion_detected=context.distortion_result.predicted_class,
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
                distortion_detected=context.distortion_result.predicted_class,
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
            and confidence >= 0.35
        )

        has_evidence = self.has_cbt_evidence(text_lower, words)

        if is_signal_present and has_evidence:
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
