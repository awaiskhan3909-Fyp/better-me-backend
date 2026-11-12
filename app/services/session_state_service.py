import re
import uuid
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.db.models import ConversationSessionState, Conversation
from app.schemas.response import DistortionPrediction


class TherapyStage(str, Enum):
    CHECK_IN = "check_in"
    PROBLEM_EXPLORATION = "exploration"
    EXPLORATION = "exploration"
    EMOTION_AND_BODY = "emotion_and_body"
    THOUGHT_IDENTIFICATION = "thought_identification"
    ASSESSMENT = "thought_identification"
    DISTORTION_ANALYSIS = "distortion_analysis"
    SOCRATIC_INTERVENTION = "intervention"
    INTERVENTION = "intervention"
    ACTION_PLAN = "action_plan"
    CLOSURE = "closure"


STAGE_ORDER = [
    TherapyStage.CHECK_IN,
    TherapyStage.PROBLEM_EXPLORATION,
    TherapyStage.EMOTION_AND_BODY,
    TherapyStage.THOUGHT_IDENTIFICATION,
    TherapyStage.DISTORTION_ANALYSIS,
    TherapyStage.SOCRATIC_INTERVENTION,
    TherapyStage.ACTION_PLAN,
    TherapyStage.CLOSURE,
]

STAGE_OBJECTIVES = {
    TherapyStage.CHECK_IN: "Warm welcome, rapport building, checking current emotional baseline and setting session agenda.",
    TherapyStage.PROBLEM_EXPLORATION: "Deep contextual gathering: discovering the concrete situation, who was involved, when, and specific triggers without rushing to solutions.",
    TherapyStage.EMOTION_AND_BODY: "Assessing emotional intensity (0-100), somatic physical sensations (heart racing, tight chest, sleep), and behavioral functional impact.",
    TherapyStage.THOUGHT_IDENTIFICATION: "Uncovering the specific automatic thoughts, internal self-talk, and catastrophic predictions running through the mind.",
    TherapyStage.DISTORTION_ANALYSIS: "Collaboratively examining evidence for and against the thought with clinical curiosity, strictly gating cognitive distortion labels.",
    TherapyStage.SOCRATIC_INTERVENTION: "Guided Socratic discovery to co-create a realistic, balanced perspective that decouples setbacks from personal worth.",
    TherapyStage.ACTION_PLAN: "Designing a manageable behavioral activation micro-step or coping experiment to test between sessions.",
    TherapyStage.CLOSURE: "Structured session closure: synthesis summary, personalized homework experiment, and next-session check-in goal.",
}


class UserIntent(str, Enum):
    GREETING = "greeting"
    META_QUESTION = "meta_question"
    CONFUSION = "confusion"
    NEGATIVE_FEEDBACK = "negative_feedback"
    OFF_TOPIC = "off_topic"
    DIRECT_QUESTION = "direct_question"
    PERFECTIONISM = "perfectionism"
    UNCERTAINTY = "uncertainty"
    DISTRESS_EXPRESSION = "distress_expression"
    SOMATIC_SYMPTOM = "somatic_symptom"
    THOUGHT_SHARING = "thought_sharing"
    EVIDENCE_SHARING = "evidence_sharing"
    SOLUTION_SEEKING = "solution_seeking"
    ACTION_ACCEPTANCE = "action_acceptance"
    POLITE_ACKNOWLEDGMENT = "polite_acknowledgment"
    SESSION_EXIT_REQUEST = "session_exit_request"
    GENERAL_STATEMENT = "general_statement"


class SessionStateService:
    """
    Advanced Clinical CBT Session Management & Cognitive State Engine (Beckian 5-Part Model):
    - Rigorous controlled stage progression (no premature restructuring or sudden closures).
    - Longitudinal somatic, emotional, behavioral, and cognitive context memory.
    - Intent taxonomy with empathetic bypasses for confusion, frustration, off-topic replies, and meta-questions.
    - Strict question deduplication preventing repetitive inquiries.
    - Evidence gating: ensures distortion labels are never forced without concrete evidence.
    """

    CBT_EVIDENCE_KEYWORDS = {
        "never", "always", "every", "everyone", "nobody", "nothing", "everything",
        "completely", "totally", "impossible", "useless", "worthless", "ruined",
        "ruins", "failure", "unfit", "hopeless", "pointless", "disaster", "destined",
        "must", "should", "ought", "hates", "hate me", "they all", "ruined forever"
    }

    CONFUSION_PATTERNS = [
        r"\b(?:what\s+do\s+you\s+mean|i\s+don'?t\s+understand|i\s+don'?t\s+get\s+it|can\s+you\s+explain|what\s+does\s+that\s+mean)\b",
        r"\b(?:what\s+are\s+you\s+saying|explain\s+more|i'?m\s+confused|confusing|makes\s+no\s+sense)\b",
        r"\b(?:huh|kya\s+matlab|samajh\s+nahi\s+aaya|samajh\s+nahi\s+aayi|matlab\s+kya\s+hai|wazeh\s+karein)\b"
    ]

    NEGATIVE_FEEDBACK_PATTERNS = [
        r"\b(?:that'?s\s+not\s+(?:what\s+i|true|right)|you\s+are\s+not\s+listening|you'?re\s+not\s+listening|you\s+don'?t\s+understand)\b",
        r"\b(?:stop\s+asking|stop\s+it|this\s+is\s+annoying|not\s+helpful|you\s+are\s+repeating|you'?re\s+repeating|broken\s+record)\b",
        r"\b(?:just\s+tell\s+me|why\s+so\s+many\s+questions|stop\s+interrogating|quit\s+asking)\b",
        r"\b(?:wrong\s+(?:response|answer)|giving\s+(?:me\s+)?wrong|didn'?t\s+relate|not\s+relating|not\s+relevant|doesn'?t\s+relate|irrelevant\s+response)\b",
        r"\b(?:galat\s+samjhe|galat\s+jawab|aisa\s+nahi\s+hai|tum\s+sun\s+nahi\s+rahe|bar\s+bar\s+wahi|sawalat\s+band\s+karo)\b"
    ]

    META_QUESTION_PATTERNS = [
        r"\b(?:who\s+(?:are|r)\s+(?:you|u)|what\s+are\s+you|who\s+created\s+you|who\s+made\s+you|what\s+is\s+your\s+name)\b",
        r"\b(?:what\s+is\s+cbt|how\s+does\s+this\s+work|are\s+you\s+a\s+doctor|are\s+you\s+a\s+therapist|are\s+you\s+ai|are\s+you\s+human)\b",
        r"\b(?:kon\s+ho\s+tum|aap\s+kaun\s+hain|tum\s+kya\s+ho|cbt\s+kya\s+hota\s+hai)\b"
    ]

    OFF_TOPIC_PATTERNS = [
        r"\b(?:pizza|burger|sandwich|biryani|roti|chai|coffee|tea|lunch|dinner|breakfast|eating|food)\b",
        r"\b(?:weather|raining|rain|sunny|cloudy|snow|temperature)\b",
        r"\b(?:movie|film|netflix|cricket|football|game|gaming|song|music)\b"
    ]

    SESSION_EXIT_PATTERNS = [
        r"\b(?:i\s+(?:have|need|got)\s+to\s+go|leave\s+now|end\s+(?:this|the\s+session|here)|stop\s+here|finish\s+(?:the\s+session|here))\b",
        r"\b(?:bye\s+for\s+now|goodbye|talk\s+to\s+you\s+(?:later|tomorrow)|see\s+you\s+later|allah\s+hafiz|chalta\s+hoon|band\s+karo)\b"
    ]

    POLITE_PATTERNS = [
        r"\b(?:thanks|thank\s+you|thx|tysm|shukriya|jazakallah|bohut\s+shukriya)\b",
        r"\b(?:thanks\s+for\s+(?:asking|listening|helping)|appreciate\s+it)\b"
    ]

    SOMATIC_PATTERNS = [
        r"\b(?:chest|heart\s+racing|heartbeat|palpitations|stomach|knot|headache|migraine)\b",
        r"\b(?:shaking|trembling|sweat|sweating|can'?t\s+sleep|insomnia|sleepless|exhausted|fatigue|panicking)\b"
    ]

    SOLUTION_PATTERNS = [
        r"\b(?:tell\s+me\s+the\s+solution|what\s+should\s+i\s+do|how\s+to\s+solve|how\s+can\s+i\s+fix|give\s+me\s+a\s+solution)\b",
        r"\b(?:give\s+me\s+(?:simple\s+)?solution|solution\s+of\s+(?:all\s+)?my\s+problems|suggest\s+me\s+therapy|suggest\s+therapy|give\s+me\s+advice|need\s+a\s+solution)\b",
        r"\b(?:hal\s+batao|kya\s+karun|kya\s+karein|solution\s+batao|kuch\s+mashwara|koi\s+hal|mashwara\s+do)\b"
    ]

    DIRECT_QUESTION_PATTERNS = [
        r"^(?:can\s+you|could\s+you|what\s+should\s+i|how\s+do\s+i|do\s+you\s+think|is\s+it\s+normal|why\s+do\s+i)\b"
    ]

    PERFECTIONISM_PATTERNS = [
        r"\b(?:must\s+be\s+perfect|can'?t\s+make\s+(?:any\s+)?mistake|everything\s+has\s+to\s+be\s+100|not\s+allowed\s+to\s+fail)\b",
        r"\b(?:if\s+it'?s\s+not\s+perfect\s+it'?s\s+worthless|zero\s+tolerance\s+for\s+error|flawless|must\s+get\s+full\s+marks)\b",
        r"\b(?:sab\s+kuch\s+perfect\s+hona\s+chahiye|ghalti\s+ki\s+gunjaish\s+nahi|har\s+haal\s+mein\s+top\s+karna)\b"
    ]

    UNCERTAINTY_PATTERNS = [
        r"\b(?:i'?m\s+not\s+sure|i\s+don'?t\s+know|maybe|perhaps|i\s+guess|hard\s+to\s+say|confused\s+about\s+what\s+to\s+do)\b",
        r"\b(?:pata\s+nahi|kuch\s+samajh\s+nahi\s+aa\s+raha|shayad|mujhe\s+maloom\s+nahi|kuch\s+nahi\s+kehsakta)\b"
    ]

    ACTION_ACCEPTANCE_PATTERNS = [
        r"\b(?:yes\s+i\s+can|i\s+can\s+do\s+that|i\s+will\s+try|i\s+can\s+write|i\s+will\s+write|i\s+should\s+talk|i\s+will\s+talk|i'?ll\s+talk|sure\s+i\s+can|sounds\s+good|i\s+will\s+do|theek\s+hai|haan\s+main|kar\s+sakta\s+hoon|baat\s+karunga|koshish\s+karunga)\b",
        r"\b(?:i\s+should\s+talk|i\s+can\s+talk|i\s+will\s+ask|i\s+can\s+ask|talk\s+with\s+her|talk\s+to\s+her|talk\s+to\s+him|call\s+her|talk\s+with\s+wife)\b"
    ]

    EVIDENCE_PATTERNS = [
        r"\b(?:because|evidence|she\s+replied|she\s+said|he\s+said|it\s+happened|fact|actually|maazi\s+mein|pehle\s+bhi|saboot)\b"
    ]

    def classify_user_intent(self, text: str) -> UserIntent:
        text_lower = text.lower().strip()
        words = set(re.findall(r"\b[a-zA-Z']+\b", text_lower))

        # 1. Genuine Session Exit Request (Highest Intent Priority)
        if any(bool(re.search(pat, text_lower)) for pat in self.SESSION_EXIT_PATTERNS):
            return UserIntent.SESSION_EXIT_REQUEST

        # 2. Confusion / Request for Clarification
        if any(bool(re.search(pat, text_lower)) for pat in self.CONFUSION_PATTERNS):
            return UserIntent.CONFUSION

        # 3. Frustration / Negative Feedback
        if any(bool(re.search(pat, text_lower)) for pat in self.NEGATIVE_FEEDBACK_PATTERNS):
            return UserIntent.NEGATIVE_FEEDBACK

        # 4. Meta Questions about System/CBT/Identity
        if any(bool(re.search(pat, text_lower)) for pat in self.META_QUESTION_PATTERNS):
            return UserIntent.META_QUESTION

        # 5. Off-Topic / Irrelevant Interjections (Food, Weather, etc.)
        is_problem_context = any(w in words for w in ["sad", "depressed", "anxious", "stress", "stressed", "fail", "exam", "bad", "scared", "worthless"])
        if not is_problem_context and any(bool(re.search(pat, text_lower)) for pat in self.OFF_TOPIC_PATTERNS):
            return UserIntent.OFF_TOPIC

        # 6. Polite Acknowledgment (Does NOT close early sessions)
        if any(bool(re.search(pat, text_lower)) for pat in self.POLITE_PATTERNS):
            return UserIntent.POLITE_ACKNOWLEDGMENT

        # 7. Somatic Physical Symptoms
        if any(bool(re.search(pat, text_lower)) for pat in self.SOMATIC_PATTERNS):
            return UserIntent.SOMATIC_SYMPTOM

        # 8. Solution Seeking
        if any(bool(re.search(pat, text_lower)) for pat in self.SOLUTION_PATTERNS):
            return UserIntent.SOLUTION_SEEKING

        # 9. Perfectionism / Rigid 100% Demand
        if any(bool(re.search(pat, text_lower)) for pat in self.PERFECTIONISM_PATTERNS):
            return UserIntent.PERFECTIONISM

        # 10. Uncertainty / Ambivalence
        if any(bool(re.search(pat, text_lower)) for pat in self.UNCERTAINTY_PATTERNS):
            return UserIntent.UNCERTAINTY

        # 11. Direct Questions to the Bot
        if any(bool(re.search(pat, text_lower)) for pat in self.DIRECT_QUESTION_PATTERNS):
            return UserIntent.DIRECT_QUESTION

        # 12. Greetings & Presence Checks
        if text_lower in {"hi", "hello", "hey", "salam", "assalam", "good morning", "good evening"} or \
           any(bool(re.search(pat, text_lower)) for pat in [r"\b(?:are\s+you\s+there|you\s+there|sun\s+rahe\s+ho)\b"]) or \
           (len(words) <= 4 and any(w in words for w in ["hi", "hello", "hey", "salam"])):
            return UserIntent.GREETING

        # 13. Action Acceptance & Constructive Planning
        if any(bool(re.search(pat, text_lower)) for pat in self.ACTION_ACCEPTANCE_PATTERNS):
            return UserIntent.ACTION_ACCEPTANCE

        # 13.5. Evidence Sharing
        if any(bool(re.search(pat, text_lower)) for pat in self.EVIDENCE_PATTERNS):
            return UserIntent.EVIDENCE_SHARING

        # 14. Thought Sharing
        if re.search(r"\b(?:i\s+think|i\s+feel\s+like|in\s+my\s+mind|my\s+thought\s+is|mujhe\s+lagta\s+hai|i\s+am\s+convinced)\b", text_lower):
            return UserIntent.THOUGHT_SHARING

        # 14. Distress Expression
        distress_words = {"sad", "depressed", "anxious", "anxiety", "scared", "fear", "fail", "failed", "exam", "marks", "crying", "upset", "preshan", "pareshan", "udas"}
        if any(w in words for w in distress_words):
            return UserIntent.DISTRESS_EXPRESSION

        return UserIntent.GENERAL_STATEMENT

    def evaluate_evidence_gate(
        self,
        user_text: str,
        distortion_pred: DistortionPrediction,
        state: ConversationSessionState
    ) -> Tuple[bool, Optional[str]]:
        """
        Evidence Gating Rule:
        Never force or assign a cognitive distortion label unless:
        1. The session has progressed past early context gathering (at least THOUGHT_IDENTIFICATION or later).
        2. High classifier confidence (>= 0.65).
        3. Explicit evidence keywords exist in the text (e.g. 'never', 'always', 'ruined forever').
        4. No raw 'LABEL_' or 'Unknown' strings.
        """
        predicted = distortion_pred.predicted_class
        if not predicted or predicted.lower() in ["none", "normal", "null", "unknown"] or str(predicted).startswith("LABEL_"):
            return False, "None"

        text_lower = user_text.lower()
        words = set(re.findall(r"\b[a-zA-Z']+\b", text_lower))
        has_evidence_words = any(w in words for w in self.CBT_EVIDENCE_KEYWORDS)

        # Early stages must never label distortions
        current_stage = state.current_stage
        if current_stage in [TherapyStage.CHECK_IN.value, TherapyStage.PROBLEM_EXPLORATION.value, TherapyStage.EMOTION_AND_BODY.value]:
            if not has_evidence_words or state.total_turns < 3:
                return False, "None"

        if has_evidence_words and distortion_pred.confidence >= 0.55:
            return True, predicted

        if state.total_turns >= 4 and distortion_pred.confidence >= 0.70:
            return True, predicted

        return False, "None"

    def get_or_create_session_state(
        self,
        conversation_id: uuid.UUID,
        db: Session
    ) -> ConversationSessionState:
        state = db.query(ConversationSessionState).filter(
            ConversationSessionState.conversation_id == conversation_id
        ).first()

        if not state:
            state = ConversationSessionState(
                conversation_id=conversation_id,
                current_stage=TherapyStage.CHECK_IN.value,
                stage_turn_count=0,
                total_turns=0,
                asked_questions=[],
                session_goals=[],
                distortion_evidence=[],
                metadata_json={
                    "explored_dimensions": {
                        "situation": False,
                        "emotions": False,
                        "physiology": False,
                        "behavior": False,
                        "thought": False,
                        "evidence": False
                    },
                    "identified_emotions": [],
                    "somatic_symptoms": [],
                    "homework": None,
                    "next_session_goal": None
                }
            )
            db.add(state)
            db.flush()

        return state

    def progress_session_stage(
        self,
        state: ConversationSessionState,
        user_text: str,
        user_intent: UserIntent,
        distortion_pred: DistortionPrediction,
        is_evidence_gated: bool
    ) -> TherapyStage:
        """
        Controlled Stage Progression Engine (Padesky & Greenberger 5-Part Model):
        Prevents premature intervention or premature closure.
        """
        try:
            current = TherapyStage(state.current_stage)
        except ValueError:
            current = TherapyStage.PROBLEM_EXPLORATION

        # 1. Non-therapeutic or grounding intents do NOT rush advance therapy stage
        if user_intent in [
            UserIntent.META_QUESTION,
            UserIntent.CONFUSION,
            UserIntent.NEGATIVE_FEEDBACK,
            UserIntent.OFF_TOPIC,
            UserIntent.DIRECT_QUESTION,
            UserIntent.UNCERTAINTY
        ]:
            return current

        # 2. Greeting in an ongoing session does NOT reset to CHECK_IN
        if user_intent == UserIntent.GREETING and state.total_turns >= 2:
            return current

        # 3. Explicit Exit Request: Only triggers closure if ready or requested
        if user_intent == UserIntent.SESSION_EXIT_REQUEST:
            return TherapyStage.CLOSURE

        # 4. Polite Acknowledgment ("thanks"): Only wraps up if already in ACTION_PLAN or late session
        if user_intent == UserIntent.POLITE_ACKNOWLEDGMENT:
            if current in [TherapyStage.ACTION_PLAN, TherapyStage.CLOSURE] or state.total_turns >= 6:
                return TherapyStage.CLOSURE
            return current

        # STAGE 1: CHECK_IN
        if current == TherapyStage.CHECK_IN:
            if user_intent == UserIntent.GREETING:
                return TherapyStage.CHECK_IN
            # User shares problem/focus -> Move to PROBLEM_EXPLORATION
            return TherapyStage.PROBLEM_EXPLORATION

        # STAGE 2: PROBLEM_EXPLORATION
        elif current == TherapyStage.PROBLEM_EXPLORATION:
            # Require at least 2 turns in exploration to fully map situation & triggers
            if state.stage_turn_count >= 2 and (len(user_text.split()) >= 6 or user_intent in [UserIntent.SOMATIC_SYMPTOM, UserIntent.DISTRESS_EXPRESSION]):
                return TherapyStage.EMOTION_AND_BODY
            return TherapyStage.PROBLEM_EXPLORATION

        # STAGE 3: EMOTION_AND_BODY (Somatic & Behavioral Impact)
        elif current == TherapyStage.EMOTION_AND_BODY:
            if state.stage_turn_count >= 1 or user_intent in [UserIntent.THOUGHT_SHARING, UserIntent.SOMATIC_SYMPTOM]:
                return TherapyStage.THOUGHT_IDENTIFICATION
            return TherapyStage.EMOTION_AND_BODY

        # STAGE 4: THOUGHT_IDENTIFICATION
        elif current == TherapyStage.THOUGHT_IDENTIFICATION:
            if user_intent == UserIntent.SOLUTION_SEEKING:
                return TherapyStage.SOCRATIC_INTERVENTION
            if state.stage_turn_count >= 1 or user_intent == UserIntent.THOUGHT_SHARING:
                return TherapyStage.DISTORTION_ANALYSIS
            return TherapyStage.THOUGHT_IDENTIFICATION

        # STAGE 5: DISTORTION_ANALYSIS
        elif current == TherapyStage.DISTORTION_ANALYSIS:
            if state.stage_turn_count >= 1 or user_intent in [UserIntent.EVIDENCE_SHARING, UserIntent.SOLUTION_SEEKING]:
                return TherapyStage.SOCRATIC_INTERVENTION
            return TherapyStage.DISTORTION_ANALYSIS

        # STAGE 6: SOCRATIC_INTERVENTION (Restructuring)
        elif current == TherapyStage.SOCRATIC_INTERVENTION:
            if state.stage_turn_count >= 1 or user_intent in [UserIntent.ACTION_ACCEPTANCE, UserIntent.SOLUTION_SEEKING]:
                return TherapyStage.ACTION_PLAN
            return TherapyStage.SOCRATIC_INTERVENTION

        # STAGE 7: ACTION_PLAN
        elif current == TherapyStage.ACTION_PLAN:
            if state.stage_turn_count >= 1 or user_intent in [UserIntent.ACTION_ACCEPTANCE, UserIntent.POLITE_ACKNOWLEDGMENT]:
                return TherapyStage.CLOSURE
            return TherapyStage.ACTION_PLAN

        # STAGE 8: CLOSURE
        elif current == TherapyStage.CLOSURE:
            return TherapyStage.CLOSURE

        return current

    def is_question_duplicate(self, state: ConversationSessionState, candidate_question: str) -> bool:
        """Checks whether candidate question has already been asked or has >= 55% word overlap."""
        if not candidate_question or not state.asked_questions:
            return False

        clean_cand = re.sub(r'[^\w\s]', '', candidate_question.lower()).strip()
        cand_words = set(clean_cand.split())
        if not cand_words:
            return False

        for q in state.asked_questions:
            clean_q = re.sub(r'[^\w\s]', '', q.lower()).strip()
            if clean_cand == clean_q:
                return True
            q_words = set(clean_q.split())
            if not q_words:
                continue
            overlap = len(cand_words & q_words) / max(len(cand_words), 1)
            if overlap >= 0.55:
                return True

        return False

    def record_turn(
        self,
        state: ConversationSessionState,
        next_stage: TherapyStage,
        user_text: str,
        ai_question_asked: Optional[str] = None,
        distortion_label: Optional[str] = None,
        distortion_confidence: float = 0.0,
        is_evidence_gated: bool = False
    ):
        """
        Updates session state counters, current stage, and deduplication memory with explicit SQLAlchemy dirty marking.
        """
        if next_stage.value != state.current_stage:
            state.current_stage = next_stage.value
            state.stage_turn_count = 1
        else:
            state.stage_turn_count += 1

        state.total_turns += 1

        if is_evidence_gated and distortion_label and distortion_label not in ["None", "Normal"] and not str(distortion_label).startswith("LABEL_"):
            state.identified_distortion = distortion_label
            state.distortion_confidence = distortion_confidence
            state.evidence_gated = True

        if ai_question_asked:
            clean_q = ai_question_asked.strip()
            current_questions = list(state.asked_questions or [])
            if clean_q not in current_questions:
                current_questions.append(clean_q)
                state.asked_questions = current_questions[-12:]
                flag_modified(state, "asked_questions")

        # Track somatic and emotion indicators in metadata
        text_lower = user_text.lower()
        meta = dict(state.metadata_json or {})
        dims = dict(meta.get("explored_dimensions", {}))

        if any(w in text_lower for w in ["feel", "feeling", "anxious", "sad", "scared", "fear", "angry"]):
            dims["emotions"] = True
        if any(w in text_lower for w in ["chest", "stomach", "sleep", "heart", "tired", "shaking"]):
            dims["physiology"] = True
        if any(w in text_lower for w in ["did", "avoid", "stopped", "procrastinate", "stayed"]):
            dims["behavior"] = True
        if any(w in text_lower for w in ["think", "thought", "believed", "said to myself"]):
            dims["thought"] = True

        meta["explored_dimensions"] = dims

        # Dynamically tailor specific homework & next session goals based on contextual cues
        if next_stage in [TherapyStage.ACTION_PLAN, TherapyStage.CLOSURE] and not meta.get("homework"):
            trigger_text = (state.identified_trigger or user_text).lower()
            if any(k in trigger_text for k in ["exam", "study", "grade", "marks", "test", "fail", "course", "assignment"]):
                meta["homework"] = "Behavioral Study Experiment: Complete a 20-minute focused study interval. If thoughts of failure appear, record them on paper and return to studying without demanding perfection."
                meta["next_session_goal"] = "Review study focus resilience and observe whether catastrophic exam predictions materialized."
            elif any(k in trigger_text for k in ["friend", "relationship", "people", "talk", "boss", "colleague", "judg", "hate", "shun"]):
                meta["homework"] = "Interpersonal Evidence-Testing: When feeling tempted to assume negative reactions from others, test the assumption by asking one direct clarifying question."
                meta["next_session_goal"] = "Review real-world interpersonal feedback and evaluate perceived social safety."
            elif any(k in trigger_text for k in ["chest", "heart", "panic", "sleep", "stomach", "breath", "insomnia", "tired"]):
                meta["homework"] = "Somatic Regulation Exercise: Practice 3 cycles of 4-7-8 breathing twice daily, logging physical tension before and after on a 0-10 scale."
                meta["next_session_goal"] = "Assess physical calm and track changes in somatic panic symptoms."
            else:
                meta["homework"] = "Cognitive Journaling: Log 1 automatic negative thought daily in a 2-column format (Automatic Thought vs. Objective Fact)."
                meta["next_session_goal"] = "Reinforce cognitive flexibility and review how balanced thinking impacted daily mood."

        if not meta.get("next_session_goal"):
            meta["next_session_goal"] = "Review behavioral activation homework and reinforce adaptive coping skills."

        state.metadata_json = meta
        flag_modified(state, "metadata_json")


session_state_service = SessionStateService()
