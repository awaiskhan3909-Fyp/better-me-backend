import re
from enum import Enum
from typing import List, Optional, Set, Any
from pydantic import BaseModel

from app.schemas.response import SafetyPrediction, DistortionPrediction, EntityItem
from app.services.conversation_manager import ConversationContext
from app.services.session_state_service import (
    session_state_service,
    TherapyStage,
    UserIntent,
    STAGE_OBJECTIVES,
)
from app.services.distortion_service import CANONICAL_DISTORTIONS


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
    session_state: Optional[Any] = None
    user_intent: Optional[str] = None


class ResponseDecision(BaseModel):
    strategy: ResponseStrategy
    priority: DecisionPriority
    reason: str
    distortion_detected: Optional[str] = None
    safety_risk_level: str = "Safe"
    decision_version: str = "v4"
    therapy_stage: Optional[str] = None
    session_objective: Optional[str] = None


class ResponseDecisionService:
    """
    Advanced Clinical Response Decision Engine (Beckian Multi-Stage Protocol):
    - Priority 1: Crisis safety override.
    - Priority 2: Direct responses for meta-questions, confusion, frustration, off-topic replies, and direct questions.
    - Priority 3: Controlled 7-Stage progression (no premature restructuring or abrupt closure).
    - Evidence Gated: Never forces cognitive distortion labels prematurely or returns raw classifier labels (LABEL_X).
    """

    CRISIS_PATTERNS = [
        r"\b(?:suicid|sucid|suicd)\w*\b",
        r"\b(?:kill|hurt|harm|cut|poison|hang|shoot|end)\s+(?:my|his|her|their|one'?s)?\s*(?:self|life)\b",
        r"\b(?:want|wants|wishing)\s+to\s+die\b",
        r"\b(?:better\s+off\s+dead|tired\s+of\s+living|no\s+reason\s+to\s+live)\b",
        r"\bend(?:ing)?\s+it\s+all\b",
        r"\bself[\s\-]harm\w*\b"
    ]

    def sanitize_distortion_label(self, raw_label: Optional[str]) -> str:
        """Guarantees that raw classifier strings like 'LABEL_3' never reach user-facing outputs."""
        if not raw_label or raw_label in ["None", "Normal", "null", "Unknown"]:
            return "None"
        label_str = str(raw_label).strip()
        if label_str.startswith("LABEL_"):
            try:
                idx = int(label_str.split("_")[-1])
                return CANONICAL_DISTORTIONS.get(idx, "None")
            except Exception:
                return "None"
        return label_str

    def evaluate_decision(self, context: DecisionContext) -> ResponseDecision:
        user_text = context.user_text.strip()
        text_lower = user_text.lower()
        state = context.session_state

        # Sanitize distortion input
        clean_distortion = self.sanitize_distortion_label(context.distortion_result.predicted_class)

        # -------------------------------------------------------------
        # RULE 1 — SAFETY FIRST (Highest Priority - Critical)
        # -------------------------------------------------------------
        safety_risk = context.safety_result.risk_level.lower()
        is_crisis_text = any(bool(re.search(pat, text_lower)) for pat in self.CRISIS_PATTERNS)

        if safety_risk in ["high risk", "high"] or context.safety_result.needs_safety_alert or is_crisis_text:
            return ResponseDecision(
                strategy=ResponseStrategy.SAFETY_RESPONSE,
                priority=DecisionPriority.CRITICAL,
                reason="high_risk_safety_alert",
                safety_risk_level="High Risk" if is_crisis_text else context.safety_result.risk_level,
                distortion_detected=clean_distortion,
                therapy_stage="safety_override",
                session_objective="Immediate crisis support, emergency hotlines, and de-escalation."
            )

        # -------------------------------------------------------------
        # RULE 2 — INTENT EVALUATION & DIRECT BYPASSES
        # -------------------------------------------------------------
        intent = session_state_service.classify_user_intent(user_text)
        current_stage_enum = TherapyStage(state.current_stage) if state else TherapyStage.CHECK_IN

        # 2A. Meta / Role / CBT Question
        if intent == UserIntent.META_QUESTION:
            return ResponseDecision(
                strategy=ResponseStrategy.NORMAL_CONVERSATION,
                priority=DecisionPriority.NORMAL,
                reason="identity_and_role_query",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage="meta_dialogue",
                session_objective="Directly answer user query about the AI's identity, role, or CBT methodology."
            )

        # 2B. Direct Confusion / Clarification Request
        if intent == UserIntent.CONFUSION:
            return ResponseDecision(
                strategy=ResponseStrategy.CLARIFICATION,
                priority=DecisionPriority.MEDIUM,
                reason="user_confusion_clarification",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage="clarification",
                session_objective="Clarify previous concept in plain conversational language without jargon."
            )

        # 2C. Direct Negative Feedback / Frustration
        if intent == UserIntent.NEGATIVE_FEEDBACK:
            return ResponseDecision(
                strategy=ResponseStrategy.NORMAL_CONVERSATION,
                priority=DecisionPriority.MEDIUM,
                reason="user_negative_feedback",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage="feedback_alignment",
                session_objective="Validate frustration with humility, pause repetitive questioning, and adapt to patient preference."
            )

        # 2D. Off-Topic / Irrelevant Interjection
        if intent == UserIntent.OFF_TOPIC:
            return ResponseDecision(
                strategy=ResponseStrategy.NORMAL_CONVERSATION,
                priority=DecisionPriority.NORMAL,
                reason="off_topic_bridge",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage=current_stage_enum.value,
                session_objective="Pleasantly acknowledge off-topic comment and gently bridge back to the session focus."
            )

        # 2E. Direct Question to Bot
        if intent == UserIntent.DIRECT_QUESTION:
            return ResponseDecision(
                strategy=ResponseStrategy.NORMAL_CONVERSATION,
                priority=DecisionPriority.NORMAL,
                reason="direct_question_answer",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage=current_stage_enum.value,
                session_objective="Directly answer patient question before resuming guided discovery."
            )

        # 2E-1. Perfectionism / Demanding 100% Standards
        if intent == UserIntent.PERFECTIONISM:
            return ResponseDecision(
                strategy=ResponseStrategy.CBT_SUPPORT,
                priority=DecisionPriority.HIGH,
                reason="perfectionism_exploration",
                distortion_detected="All-or-Nothing Thinking",
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage=current_stage_enum.value if current_stage_enum != TherapyStage.CHECK_IN else TherapyStage.PROBLEM_EXPLORATION.value,
                session_objective="Address rigid perfectionistic standards, explore the emotional cost of demanding 100%, and introduce the 'good enough' principle."
            )

        # 2E-2. Direct Uncertainty / Ambivalence
        if intent == UserIntent.UNCERTAINTY:
            return ResponseDecision(
                strategy=ResponseStrategy.EXPLORATORY_FOLLOW_UP,
                priority=DecisionPriority.MEDIUM,
                reason="uncertainty_grounding",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage=current_stage_enum.value,
                session_objective="Validate uncertainty with warmth, remove performance pressure, and offer gentle grounding."
            )

        # 2F. Greeting Handling:
        # If early in session (turn <= 1) -> CHECK_IN
        # If mid-session (turns >= 2) -> Acknowledge continuity WITHOUT restarting intro!
        if intent == UserIntent.GREETING:
            if state and state.total_turns >= 2:
                return ResponseDecision(
                    strategy=ResponseStrategy.NORMAL_CONVERSATION,
                    priority=DecisionPriority.NORMAL,
                    reason="mid_session_greeting_acknowledgment",
                    distortion_detected="None",
                    safety_risk_level=context.safety_result.risk_level,
                    therapy_stage=current_stage_enum.value,
                    session_objective="Acknowledge presence and seamlessly continue ongoing session without restarting introduction."
                )
            return ResponseDecision(
                strategy=ResponseStrategy.NORMAL_CONVERSATION,
                priority=DecisionPriority.NORMAL,
                reason="casual_conversation_or_greeting",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage=TherapyStage.CHECK_IN.value,
                session_objective=STAGE_OBJECTIVES[TherapyStage.CHECK_IN]
            )

        # 2G. Explicit Session Exit Request
        if intent == UserIntent.SESSION_EXIT_REQUEST:
            return ResponseDecision(
                strategy=ResponseStrategy.NORMAL_CONVERSATION,
                priority=DecisionPriority.NORMAL,
                reason="session_summary_and_closure",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage=TherapyStage.CLOSURE.value,
                session_objective=STAGE_OBJECTIVES[TherapyStage.CLOSURE]
            )

        # 2H. Polite Acknowledgment ("thanks"):
        # If in ACTION_PLAN or late session -> Can close
        # If early in session -> Validate rapport, DO NOT CLOSE SESSION!
        if intent == UserIntent.POLITE_ACKNOWLEDGMENT:
            if current_stage_enum in [TherapyStage.ACTION_PLAN, TherapyStage.CLOSURE] or (state and state.total_turns >= 6):
                return ResponseDecision(
                    strategy=ResponseStrategy.NORMAL_CONVERSATION,
                    priority=DecisionPriority.NORMAL,
                    reason="session_summary_and_closure",
                    distortion_detected="None",
                    safety_risk_level=context.safety_result.risk_level,
                    therapy_stage=TherapyStage.CLOSURE.value,
                    session_objective=STAGE_OBJECTIVES[TherapyStage.CLOSURE]
                )
            # Early session: continue therapy!
            return ResponseDecision(
                strategy=ResponseStrategy.EXPLORATORY_FOLLOW_UP,
                priority=DecisionPriority.NORMAL,
                reason="polite_acknowledgment_continue",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage=current_stage_enum.value,
                session_objective="Warmly acknowledge appreciation and continue exploring the patient's situation."
            )

        # -------------------------------------------------------------
        # RULE 3 — CONTROLLED MULTI-STAGE CBT SESSION FLOW
        # -------------------------------------------------------------
        is_evidence_gated = False
        gated_distortion = "None"
        if state:
            is_evidence_gated, gated_distortion = session_state_service.evaluate_evidence_gate(
                user_text, context.distortion_result, state
            )
            gated_distortion = self.sanitize_distortion_label(gated_distortion)

            next_stage = session_state_service.progress_session_stage(
                state=state,
                user_text=user_text,
                user_intent=intent,
                distortion_pred=context.distortion_result,
                is_evidence_gated=is_evidence_gated
            )
        else:
            next_stage = TherapyStage.PROBLEM_EXPLORATION

        stage_obj = STAGE_OBJECTIVES.get(next_stage, "Supportive CBT therapeutic dialogue.")

        # Stage 1: CHECK_IN
        if next_stage == TherapyStage.CHECK_IN:
            return ResponseDecision(
                strategy=ResponseStrategy.NORMAL_CONVERSATION,
                priority=DecisionPriority.NORMAL,
                reason="session_check_in_and_rapport",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage=next_stage.value,
                session_objective=stage_obj
            )

        # Stage 2: PROBLEM_EXPLORATION (Deep Situational Context)
        elif next_stage == TherapyStage.PROBLEM_EXPLORATION:
            return ResponseDecision(
                strategy=ResponseStrategy.EXPLORATORY_FOLLOW_UP,
                priority=DecisionPriority.NORMAL,
                reason="problem_exploration_and_context",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage=next_stage.value,
                session_objective=stage_obj
            )

        # Stage 3: EMOTION_AND_BODY (Somatic, Emotional, Behavioral Impact)
        elif next_stage == TherapyStage.EMOTION_AND_BODY:
            return ResponseDecision(
                strategy=ResponseStrategy.EXPLORATORY_FOLLOW_UP,
                priority=DecisionPriority.NORMAL,
                reason="emotion_and_somatic_assessment",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage=next_stage.value,
                session_objective=stage_obj
            )

        # Stage 4: THOUGHT_IDENTIFICATION (Automatic Thoughts)
        elif next_stage == TherapyStage.THOUGHT_IDENTIFICATION:
            return ResponseDecision(
                strategy=ResponseStrategy.EXPLORATORY_FOLLOW_UP,
                priority=DecisionPriority.NORMAL,
                reason="automatic_thought_identification",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage=next_stage.value,
                session_objective=stage_obj
            )

        # Stage 5: DISTORTION_ANALYSIS (Collaborative Evidence Testing)
        elif next_stage == TherapyStage.DISTORTION_ANALYSIS:
            dist_label = gated_distortion if is_evidence_gated else clean_distortion
            return ResponseDecision(
                strategy=ResponseStrategy.CBT_SUPPORT,
                priority=DecisionPriority.HIGH,
                reason="distortion_analysis_evidence_testing",
                distortion_detected=dist_label,
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage=next_stage.value,
                session_objective=stage_obj
            )

        # Stage 6: SOCRATIC_INTERVENTION (Balanced Cognitive Restructuring)
        elif next_stage == TherapyStage.SOCRATIC_INTERVENTION:
            dist_label = gated_distortion if is_evidence_gated else clean_distortion
            return ResponseDecision(
                strategy=ResponseStrategy.CBT_SUPPORT,
                priority=DecisionPriority.HIGH,
                reason="socratic_cognitive_restructuring",
                distortion_detected=dist_label,
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage=next_stage.value,
                session_objective=stage_obj
            )

        # Stage 7: ACTION_PLAN (Behavioral Activation Micro-Step)
        elif next_stage == TherapyStage.ACTION_PLAN:
            return ResponseDecision(
                strategy=ResponseStrategy.CBT_SUPPORT,
                priority=DecisionPriority.HIGH,
                reason="personalized_action_planning",
                distortion_detected=gated_distortion if is_evidence_gated else "None",
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage=next_stage.value,
                session_objective=stage_obj
            )

        # Stage 8: CLOSURE (Summary, Homework, Next Session Goal)
        elif next_stage == TherapyStage.CLOSURE:
            return ResponseDecision(
                strategy=ResponseStrategy.NORMAL_CONVERSATION,
                priority=DecisionPriority.NORMAL,
                reason="session_summary_and_closure",
                distortion_detected="None",
                safety_risk_level=context.safety_result.risk_level,
                therapy_stage=next_stage.value,
                session_objective=stage_obj
            )

        # Default fallback
        return ResponseDecision(
            strategy=ResponseStrategy.EXPLORATORY_FOLLOW_UP,
            priority=DecisionPriority.NORMAL,
            reason="supportive_dialogue",
            distortion_detected="None",
            safety_risk_level=context.safety_result.risk_level,
            therapy_stage=TherapyStage.PROBLEM_EXPLORATION.value,
            session_objective=STAGE_OBJECTIVES[TherapyStage.PROBLEM_EXPLORATION]
        )


response_decision_service = ResponseDecisionService()
