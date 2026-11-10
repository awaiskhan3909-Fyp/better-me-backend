from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from app.services.conversation_manager import ConversationContext
from app.schemas.response import DistortionPrediction, EntityItem, CBTGuidance

PROMPT_VERSION = "better_me_cbt_session_v3"

SYSTEM_INSTRUCTION_V3 = """
ROLE & IDENTITY:
You are Better Me, a compassionate, clinically-grounded AI CBT (Cognitive Behavioral Therapy) companion.
You guide users through structured, evidence-based cognitive restructuring using guided discovery (the Socratic method) to foster resilience and clarity.

CORE CONVERSATIONAL PRINCIPLES:
1. SINGLE QUESTION RULE:
   - Ask at most ONE clear, open, thoughtful question at the end of your response.
   - NEVER ask two or three questions at once. Bombarding the user causes cognitive fatigue.
   - If providing a session summary, closing affirmation, or action plan, you may ask zero questions.

2. AVOID GENERIC EMPATHY SCRIPTS:
   - Do NOT repeat canned filler like "I hear how hard that is" or "It is completely valid to feel sad" every single turn.
   - Instead, react naturally, authentically, and contextually to the specific detail the user just shared.

3. CONTEXT AWARENESS & ADAPTABILITY:
   - Read the user's actual words carefully. If they ask a direct question about you, express confusion, or give negative feedback, address it directly and honestly before continuing the session.
   - Do NOT force a CBT cognitive distortion diagnosis on a single sentence. Only explore distortions once the concrete situation and automatic thoughts are uncovered.

4. CLEAN DELIVERY & BOUNDARIES:
   - You are a supportive AI companion, not an emergency psychiatrist or human doctor.
   - NEVER output internal annotations, stage labels (e.g. "(Socratic Question)", "(Validation)"), prompt instructions, or evaluation commentary.
   - Keep responses focused, warm, and concise (typically 2 to 4 clear sentences).
""".strip()


class LLMInputContract(BaseModel):
    user_text: str
    strategy: str
    decision_reason: str
    safety_risk_level: str
    therapy_stage: Optional[str] = "exploration"
    session_objective: Optional[str] = None
    distortion: Optional[Dict[str, Any]] = None
    entities: List[str] = []
    cbt_grounding: Optional[Dict[str, Any]] = None
    recent_history: List[Dict[str, str]] = []
    asked_questions: List[str] = []
    clinical_memory: Optional[str] = None
    is_roman_urdu: bool = False


class PromptBuilder:
    """
    Builds structured, multi-stage CBT session prompts for Better Me LLM execution.
    """

    def __init__(self, prompt_version: str = PROMPT_VERSION):
        self.prompt_version = prompt_version
        self.system_instruction = SYSTEM_INSTRUCTION_V3

    def build_contract(
        self,
        user_text: str,
        strategy: str,
        decision_reason: str,
        safety_risk_level: str,
        therapy_stage: Optional[str] = None,
        session_objective: Optional[str] = None,
        distortion_pred: Optional[DistortionPrediction] = None,
        entities: Optional[List[EntityItem]] = None,
        cbt_guidance: Optional[CBTGuidance] = None,
        conversation_context: Optional[ConversationContext] = None,
        asked_questions: Optional[List[str]] = None,
        clinical_memory: Optional[str] = None,
        is_roman_urdu: bool = False
    ) -> LLMInputContract:
        # Extract recent turns (last 6 turns for optimal context window)
        history_list = []
        if conversation_context and conversation_context.recent_messages:
            for msg in conversation_context.recent_messages[-6:]:
                sender = "user" if msg.sender_type == "user" else "assistant"
                history_list.append({"sender": sender, "content": msg.content})

        distortion_dict = None
        if distortion_pred and distortion_pred.predicted_class not in ["None", "Normal", "no distortion"]:
            distortion_dict = {
                "label": distortion_pred.predicted_class,
                "confidence": distortion_pred.confidence
            }

        entities_list = [f"{e.text} ({e.label})" for e in entities] if entities else []
        cbt_dict = cbt_guidance.model_dump() if cbt_guidance else None

        return LLMInputContract(
            user_text=user_text,
            strategy=strategy,
            decision_reason=decision_reason,
            safety_risk_level=safety_risk_level,
            therapy_stage=therapy_stage or "exploration",
            session_objective=session_objective,
            distortion=distortion_dict,
            entities=entities_list,
            cbt_grounding=cbt_dict,
            recent_history=history_list,
            asked_questions=asked_questions or [],
            clinical_memory=clinical_memory,
            is_roman_urdu=is_roman_urdu
        )

    def build_prompt_text(self, contract: LLMInputContract) -> str:
        prompt_parts = []

        # 0. Longitudinal Patient Memory (Clinical Continuity)
        if contract.clinical_memory:
            prompt_parts.append(contract.clinical_memory)
            prompt_parts.append("")

        # 1. Recent Conversation History
        if contract.recent_history:
            prompt_parts.append("--- Recent Conversation History ---")
            for msg in contract.recent_history:
                role_label = "USER" if msg["sender"] == "user" else "ASSISTANT"
                prompt_parts.append(f"{role_label}: {msg['content']}")
            prompt_parts.append("")

        # 2. Previously Asked Questions (To prevent repetition)
        if contract.asked_questions:
            prompt_parts.append("--- Questions Already Asked in this Session (DO NOT REPEAT THESE) ---")
            for q in contract.asked_questions[-4:]:
                prompt_parts.append(f"- {q}")
            prompt_parts.append("")

        # 3. Current User Input
        prompt_parts.append("--- Current User Message ---")
        prompt_parts.append(f"USER: {contract.user_text}")

        # 4. Context & Stage Directives
        prompt_parts.append("\n--- Session State & Strategy Guidance ---")
        prompt_parts.append(f"Active Therapy Stage: {contract.therapy_stage}")
        if contract.session_objective:
            prompt_parts.append(f"Stage Objective: {contract.session_objective}")

        if contract.distortion and contract.therapy_stage in ["distortion_analysis", "intervention"]:
            prompt_parts.append(f"Candidate Thought Pattern: {contract.distortion['label']} (Confidence: {contract.distortion['confidence']:.2f})")

        if contract.entities:
            prompt_parts.append(f"Key Situational Details: {', '.join(contract.entities)}")

        # 5. Stage-Specific Clinical Directives
        prompt_parts.append("\n--- Execution Directive ---")
        stage = contract.therapy_stage

        if contract.decision_reason == "identity_and_role_query":
            prompt_parts.append(
                "DIRECTIVE (DIRECT META-QUERY RESPONSE):\n"
                "The user is asking about your identity, role, or how you work (e.g., 'Who are you?', 'What is CBT?').\n"
                "- Answer directly, warmly, and concisely in 2-3 sentences.\n"
                "- Explain that you are Better Me, an AI CBT companion built to help explore and reframe difficult thoughts.\n"
                "- Ask one gentle question inviting them to share what brought them in today."
            )
        elif contract.decision_reason == "user_confusion_clarification":
            prompt_parts.append(
                "DIRECTIVE (CLARIFICATION):\n"
                "The user expressed confusion or asked what you meant.\n"
                "- Clarify your previous point in simple, everyday language without therapeutic jargon.\n"
                "- Ask one gentle check-in question to ensure they feel understood."
            )
        elif contract.decision_reason == "user_negative_feedback":
            prompt_parts.append(
                "DIRECTIVE (FEEDBACK ACKNOWLEDGMENT & DE-ESCALATION):\n"
                "The user expressed frustration or told you to stop asking questions.\n"
                "- Acknowledge their frustration with genuine humility and zero defensiveness.\n"
                "- Immediately pause repetitive questions.\n"
                "- Offer a direct summary or ask how they would prefer to proceed."
            )
        elif contract.decision_reason == "off_topic_bridge":
            prompt_parts.append(
                "DIRECTIVE (OFF-TOPIC CONVERSATIONAL BRIDGE):\n"
                "The user made an off-topic or casual remark (e.g. food, weather, hobbies).\n"
                "- Acknowledge their comment warmly in one brief sentence.\n"
                "- Gently bridge back to the ongoing session topic without judging them."
            )
        elif contract.decision_reason == "direct_question_answer":
            prompt_parts.append(
                "DIRECTIVE (DIRECT QUESTION ANSWERING):\n"
                "The user asked a direct question (e.g. 'What should I do?', 'Can CBT cure anxiety?').\n"
                "- Answer their question directly and honestly in 2 sentences.\n"
                "- Follow up with one focused exploratory question."
            )
        elif contract.decision_reason == "mid_session_greeting_acknowledgment":
            prompt_parts.append(
                "DIRECTIVE (MID-SESSION CONTINUITY):\n"
                "The user said hello or hey in the middle of the session.\n"
                "- DO NOT restart your introduction or say 'Hello, I am Better Me'!\n"
                "- Acknowledge their presence and smoothly pick up the previous conversation thread."
            )
        elif contract.decision_reason == "polite_acknowledgment_continue":
            prompt_parts.append(
                "DIRECTIVE (EARLY GRATITUDE ACKNOWLEDGMENT):\n"
                "The user said 'thanks' or 'thank you' during exploration.\n"
                "- DO NOT close or end the session!\n"
                "- Warmly acknowledge their gratitude in 1 sentence, and continue exploring the situation."
            )
        elif contract.decision_reason == "session_summary_and_closure" or stage == "closure":
            prompt_parts.append(
                "DIRECTIVE (STAGE 8 — STRUCTURED SESSION CLOSURE):\n"
                "The session is concluding.\n"
                "Provide a structured, clinically organized closure containing:\n"
                "1. Summary: A concise recap of the core situation, automatic thought, and balanced reframe discovered today.\n"
                "2. Personalized Homework: One practical behavioral experiment or self-reflection exercise to practice before the next session.\n"
                "3. Next-Session Objective: What you will review together next time.\n"
                "DO NOT ask any questions that restart the session!"
            )
        elif stage == "check_in":
            prompt_parts.append(
                "DIRECTIVE (STAGE 1 — SESSION OPENING & MOOD CHECK-IN):\n"
                "- Greet the user warmly and create a welcoming, non-judgmental space.\n"
                "- Ask ONE open question to check their current emotional baseline or set today's focus."
            )
        elif stage in ["exploration", "problem_exploration"]:
            prompt_parts.append(
                "DIRECTIVE (STAGE 2 — DEEP PROBLEM EXPLORATION & CONTEXT GATHERING):\n"
                "- Thoroughly explore the concrete situation: who was there, when did it happen, and what specific trigger set it off.\n"
                "- DO NOT jump to solutions, advice, or cognitive restructuring yet!\n"
                "- Ask exactly ONE open-ended exploratory question."
            )
        elif stage == "emotion_and_body":
            prompt_parts.append(
                "DIRECTIVE (STAGE 3 — EMOTIONS, PHYSIOLOGY & BEHAVIORAL IMPACT):\n"
                "- Inquire about emotional intensity, physical body sensations (racing heart, tight chest, stomach knot, insomnia), and behavioral impact (avoiding, procrastinating, isolating).\n"
                "- Ask exactly ONE question exploring the physical sensations or emotional intensity."
            )
        elif stage in ["thought_identification", "assessment"]:
            prompt_parts.append(
                "DIRECTIVE (STAGE 4 — AUTOMATIC THOUGHT IDENTIFICATION):\n"
                "- Help the user identify their exact automatic thought or internal narrative (e.g. 'When that happened, what specific thought ran through your mind?').\n"
                "- Ask exactly ONE focused question to uncover the underlying automatic thought."
            )
        elif stage == "distortion_analysis":
            prompt_parts.append(
                "DIRECTIVE (STAGE 5 — EVIDENCE TESTING & DISTORTION ANALYSIS):\n"
                "- Collaborate with the user to examine factual evidence supporting and contradicting their thought.\n"
                "- Never force a diagnostic label on them.\n"
                "- Ask ONE Socratic question testing facts versus assumptions."
            )
        elif stage in ["intervention", "socratic_intervention"]:
            prompt_parts.append(
                "DIRECTIVE (STAGE 6 — SOCRATIC RESTRUCTURING & BALANCED PERSPECTIVE):\n"
                "- Guide the user toward discovering a realistic, balanced thought that decouples setbacks from personal worth.\n"
                "- Ask ONE reflective Socratic question that helps reframe the situation constructively."
            )
            if contract.cbt_grounding and contract.cbt_grounding.get("balanced_thought_guidance"):
                prompt_parts.append(f"CBT Grounding Hint: {contract.cbt_grounding['balanced_thought_guidance']}")
        elif stage == "action_plan":
            prompt_parts.append(
                "DIRECTIVE (STAGE 7 — PERSONALIZED ACTION PLANNING):\n"
                "- Help the user define ONE concrete, manageable behavioral micro-step or coping experiment (e.g. 2-minute breathing, writing down 3 objective facts, or taking a short pause).\n"
                "- Ask if that micro-step feels achievable for them today."
            )
        else:
            prompt_parts.append(
                "DIRECTIVE: Respond empathetically, naturally, and ask exactly ONE relevant follow-up question."
            )

        # 6. Language Localization Directive (Roman Urdu)
        if contract.is_roman_urdu:
            prompt_parts.append(
                "\n--- LANGUAGE DIRECTIVE: ROMAN URDU ---\n"
                "The user is speaking in Roman Urdu (Urdu written in Latin letters).\n"
                "Write your entire response in warm, natural, comforting Roman Urdu (e.g., 'Main samajh sakta hoon...', 'Aapke khayal mein...').\n"
                "Do NOT use English. Ensure exactly ONE clear question at the end."
            )

        prompt_parts.append("\nGenerate your response now:")
        return "\n".join(prompt_parts)


prompt_builder = PromptBuilder()
