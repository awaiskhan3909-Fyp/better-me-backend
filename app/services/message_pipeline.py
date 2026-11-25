import uuid
import re
import logging
from datetime import datetime, timezone
from typing import Optional
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.db.models import Conversation
from app.db.repositories.conversation_repository import ConversationRepository
from app.db.repositories.message_repository import MessageRepository
from app.db.repositories.analysis_repository import AnalysisRepository
from app.db.repositories.ai_response_repository import AIResponseRepository

from app.schemas.response import (
    MessageDetailResponse,
    ConversationMessageResponse,
    AnalyzeResponse,
    AIResponseDetailResponse,
    ResponseDecisionDetail,
    CBTGuidance,
    DistortionPrediction,
)

from app.services.safety_service import safety_service
from app.services.distortion_service import distortion_service
from app.services.ner_service import ner_service
from app.services.cbt_service import cbt_service
from app.services.conversation_manager import conversation_manager
from app.services.response_decision_service import (
    response_decision_service,
    DecisionContext,
    ResponseStrategy,
    ResponseDecision,
)
from app.services.llm_service import llm_service, LLMResult
from app.services.clinical_memory_service import clinical_memory_service
from app.services.language_service import language_service
from app.services.session_state_service import session_state_service, TherapyStage
from app.services.llm.response_parser import response_parser

logger = logging.getLogger("better_me.message_pipeline")


class MessagePipeline:
    """
    Core CBT Conversational Pipeline.
    Strict pipeline sequence:
      1. Context & Session State retrieval
      2. Language normalization (Roman Urdu support)
      3. Safety classifier + deterministic crisis rules (BEFORE any LLM call)
      4. Distortion classifier, NER, CBT guidance
      5. Response decision engine (strategy + stage)
      6. Clinical memory briefing
      7. LLM response generation (Gemini, with offline fallback)
      8. State update, evidence gating, telemetry, single atomic transaction commit
    """

    def process(
        self,
        conversation_id: uuid.UUID,
        text: str,
        db: Session,
        conversation: Optional[Conversation] = None
    ) -> ConversationMessageResponse:
        conv_repo = ConversationRepository(db)
        msg_repo = MessageRepository(db)
        analysis_repo = AnalysisRepository(db)
        ai_response_repo = AIResponseRepository(db)

        if conversation is None:
            conversation = conv_repo.get_by_id(conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found.")

        try:
            # Step 1: Retrieve recent conversation context & session state
            conv_context = conversation_manager.get_conversation_context(
                conversation_id=conversation_id,
                db=db,
                current_text=text
            )
            session_state = session_state_service.get_or_create_session_state(
                conversation_id=conversation_id,
                db=db
            )

            # Auto-title conversation with descriptive topic if currently generic
            curr_title = (conversation.title or "").strip()
            if curr_title in ["New Session", "Therapy Session"] or curr_title.startswith("Session "):
                t_low = text.lower()
                if any(k in t_low for k in ["exam", "study", "marks", "grade", "test", "fail", "course", "assignment"]):
                    conversation.title = "Academic & Exam Stress"
                elif any(k in t_low for k in ["work", "meeting", "presentation", "boss", "manager", "job", "office", "colleague"]):
                    conversation.title = "Workplace Performance Worry"
                elif any(k in t_low for k in ["friend", "relationship", "people", "hate me", "lonely", "family", "partner"]):
                    conversation.title = "Interpersonal Reflection"
                elif any(k in t_low for k in ["heart", "chest", "breath", "panic", "shaking", "sleep", "insomnia"]):
                    conversation.title = "Somatic Anxiety Relief"
                elif any(k in t_low for k in ["perfect", "mistake", "100%", "worthless"]):
                    conversation.title = "Overcoming Perfectionism"
                else:
                    words = [w for w in re.sub(r"[^\w\s]", "", text).split() if len(w) > 2]
                    if len(words) >= 2:
                        conversation.title = " ".join(words[:4]).title()
                    else:
                        conversation.title = "CBT Therapy Reflection"

            # Step 2: Create User Message
            user_message = msg_repo.add_message(
                conversation_id=conversation_id,
                sender_type="user",
                content=text,
                commit=False
            )

            # Step 2.5: Roman Urdu & Language Intelligence Normalization
            lang_result = language_service.process_input(text)
            clinical_text = lang_result.english_text if lang_result.is_roman_urdu else text

            # Step 3: Run prediction models (Safety FIRST, then Distortion, NER, CBT Guidance)
            # Using clinical_text for BERT classifiers to guarantee high accuracy
            safety_pred = safety_service.predict(
                clinical_text,
                is_vernacular_crisis=lang_result.is_vernacular_crisis
            )
            distortion_pred = distortion_service.predict(clinical_text)
            entities = ner_service.extract_entities(clinical_text)
            cbt_guidance = cbt_service.get_cbt_guidance(clinical_text, distortion_pred.predicted_class)

            # Step 4: Evaluate Response Strategy via Response Decision Engine
            decision_context = DecisionContext(
                user_text=text,
                safety_result=safety_pred,
                distortion_result=distortion_pred,
                entities=entities,
                conversation_context=conv_context,
                session_state=session_state
            )
            decision: ResponseDecision = response_decision_service.evaluate_decision(decision_context)

            # Step 4.5: Longitudinal Clinical & Cross-Session Memory Context Extraction
            clinical_memory_briefing = clinical_memory_service.build_clinical_memory_briefing(
                user_id=conversation.user_id,
                current_text=text,
                db=db,
                current_conversation_id=conversation_id
            )

            # Step 5: Route Response Strategy to Conversational LLM Service
            default_cbt_guidance = None
            if decision.strategy == ResponseStrategy.CBT_SUPPORT:
                default_cbt_guidance = cbt_service.get_cbt_guidance(clinical_text, distortion_pred.predicted_class)

            llm_result: LLMResult = llm_service.generate_response(
                user_text=text,
                strategy=decision.strategy.value,
                decision_reason=decision.reason,
                safety_risk_level=safety_pred.risk_level,
                distortion_pred=distortion_pred,
                entities=entities,
                cbt_guidance=default_cbt_guidance,
                conversation_context=conv_context,
                clinical_memory=clinical_memory_briefing,
                is_roman_urdu=lang_result.is_roman_urdu,
                therapy_stage=decision.therapy_stage,
                session_objective=decision.session_objective,
                asked_questions=session_state.asked_questions if session_state else []
            )

            ai_message_content = llm_result.content
            response_source = llm_result.response_source
            response_type = llm_result.response_type
            model_name = llm_result.model_name
            cbt_data_json = llm_result.cbt_data
            llm_metadata_json = llm_result.llm_metadata or {}

            # Update structured CBT session state
            ai_question = response_parser.extract_question(ai_message_content)
            stage_val = decision.therapy_stage
            valid_stages = {s.value for s in TherapyStage}
            if stage_val in valid_stages:
                target_stage = TherapyStage(stage_val)
            elif session_state and session_state.current_stage in valid_stages:
                target_stage = TherapyStage(session_state.current_stage)
            else:
                target_stage = TherapyStage.PROBLEM_EXPLORATION

            session_state_service.record_turn(
                state=session_state,
                next_stage=target_stage,
                user_text=text,
                ai_question_asked=ai_question,
                distortion_label=decision.distortion_detected,
                distortion_confidence=distortion_pred.confidence,
                is_evidence_gated=(decision.distortion_detected not in ["None", "Normal", None])
            )

            # Turn Execution Telemetry Logging (Zero print statements)
            safety_conf = (getattr(safety_pred, "confidence", None) or safety_pred.probabilities.get(safety_pred.risk_level, 0.98)) * 100
            ner_tags = ", ".join([f"{e.label}: {e.text}" for e in entities]) if entities else "None"
            latency = llm_metadata_json.get("latency_ms", "N/A")
            latency_str = f"{latency:.0f} ms" if isinstance(latency, (int, float)) else str(latency)

            logger.info(
                f"Turn Telemetry | Input: '{text[:90]}' | "
                f"Safety: {safety_pred.risk_level} ({safety_conf:.1f}%) | "
                f"Distortion: {distortion_pred.predicted_class} ({distortion_pred.confidence * 100:.1f}%) | "
                f"Stage: {decision.therapy_stage} ({decision.reason}) | "
                f"NER: [{ner_tags}] | Model: {model_name} ({response_source}) | Latency: {latency_str}"
            )
            if llm_metadata_json.get("fallback_used"):
                logger.warning(f"Turn Telemetry Fallback Triggered | Reason: {llm_metadata_json.get('fallback_reason')}")

            # Configure response schemas for frontend/analysis contract (Evidence Gated)
            clean_pred_class = response_decision_service.sanitize_distortion_label(distortion_pred.predicted_class)
            clean_detected = response_decision_service.sanitize_distortion_label(decision.distortion_detected)

            is_distortion_gated = (
                clean_detected
                and clean_detected not in ["None", "Normal", "null"]
                and not clean_detected.startswith("LABEL_")
                and decision.strategy == ResponseStrategy.CBT_SUPPORT
            )

            # Sanitize all_probabilities keys to never contain LABEL_
            sanitized_probs = {}
            for k, v in distortion_pred.all_probabilities.items():
                clean_k = response_decision_service.sanitize_distortion_label(k)
                sanitized_probs[clean_k] = v

            if not is_distortion_gated:
                distortion_output = DistortionPrediction(
                    predicted_class="None",
                    confidence=distortion_pred.confidence if clean_detected == "None" else 0.999,
                    all_probabilities=sanitized_probs
                )
                cbt_guidance = CBTGuidance(
                    detected_distortion="None",
                    template_id=None,
                    title=None,
                    explanation=None,
                    reframing_question=None,
                    balanced_thought_guidance=ai_message_content,
                    small_action=None
                )
            elif decision.strategy == ResponseStrategy.SAFETY_RESPONSE:
                cbt_guidance = CBTGuidance(
                    detected_distortion="Safety Intervention",
                    template_id=None,
                    title="Safety Support",
                    explanation=None,
                    reframing_question=None,
                    balanced_thought_guidance=ai_message_content,
                    small_action=None
                )
                distortion_output = DistortionPrediction(
                    predicted_class="None",
                    confidence=distortion_pred.confidence,
                    all_probabilities=sanitized_probs
                )
            else:
                final_class = clean_detected if clean_detected not in ["None", "Normal"] else clean_pred_class
                cbt_guidance = CBTGuidance(
                    detected_distortion=final_class,
                    template_id=None,
                    title=f"Cognitive Distortion: {final_class}",
                    explanation=None,
                    reframing_question=None,
                    balanced_thought_guidance=ai_message_content,
                    small_action=None
                )
                distortion_output = DistortionPrediction(
                    predicted_class=final_class,
                    confidence=distortion_pred.confidence,
                    all_probabilities=sanitized_probs
                )

            # Step 6: Create MessageAnalysis
            entities_data = [e.model_dump() for e in entities]
            analysis = analysis_repo.create_analysis(
                message_id=user_message.id,
                conversation_id=conversation_id,
                safety_risk_level=safety_pred.risk_level,
                needs_safety_alert=safety_pred.needs_safety_alert,
                safety_probabilities=safety_pred.probabilities,
                distortion_class=distortion_pred.predicted_class,
                distortion_confidence=distortion_pred.confidence,
                distortion_probabilities=distortion_pred.all_probabilities,
                entities=entities_data,
                model_metadata={
                    "distortion_model": "bert_cognitive_distortions",
                    "safety_model": "bert_safety_classifier",
                    "ner_model": "custom_ner_model"
                },
                commit=False
            )

            # Step 7: Create AI Message
            ai_message = msg_repo.add_message(
                conversation_id=conversation_id,
                sender_type="ai",
                content=ai_message_content,
                commit=False
            )

            # Step 8: Create AIResponse record with LLM metadata & Decision metadata
            ai_response_log = ai_response_repo.create_ai_response(
                conversation_id=conversation_id,
                user_message_id=user_message.id,
                ai_message_id=ai_message.id,
                response_source=response_source,
                response_type=response_type,
                model_name=model_name,
                response_content=ai_message_content,
                cbt_data=cbt_data_json,
                llm_metadata=llm_metadata_json,
                metadata_json={
                    "decision_strategy": decision.strategy.value,
                    "decision_priority": decision.priority.value,
                    "decision_reason": decision.reason,
                    "decision_version": decision.decision_version,
                    "therapy_stage": decision.therapy_stage,
                    "session_objective": decision.session_objective,
                },
                commit=False
            )

            # Step 9: Update Conversation current_risk_level and updated_at timestamp
            conv_repo.update_risk_level(conversation_id, safety_pred.risk_level, commit=False)
            conversation.updated_at = datetime.now(timezone.utc)

            # Step 10: Single atomic transaction commit
            db.commit()

            # Step 11: Refresh DB objects
            db.refresh(user_message)
            db.refresh(ai_message)
            db.refresh(ai_response_log)

            # Step 12: Record Turn Insights into Longitudinal Clinical Memory
            try:
                clinical_memory_service.record_turn_insight(
                    user_id=conversation.user_id,
                    conversation_id=conversation_id,
                    user_text=text,
                    detected_distortion=distortion_pred.predicted_class,
                    reframe_text=ai_message_content,
                    db=db
                )
            except Exception:
                pass

        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while processing message: {str(e)}"
            )

        analyze_res = AnalyzeResponse(
            text=text,
            safety=safety_pred,
            distortion=distortion_output,
            entities=entities,
            cbt_guidance=cbt_guidance
        )

        decision_detail = ResponseDecisionDetail(
            strategy=decision.strategy.value,
            priority=decision.priority.value,
            reason=decision.reason,
            distortion_detected=decision.distortion_detected,
            safety_risk_level=decision.safety_risk_level,
            decision_version=decision.decision_version,
            therapy_stage=decision.therapy_stage,
            session_objective=decision.session_objective
        )

        return ConversationMessageResponse(
            conversation_id=conversation_id,
            user_message=MessageDetailResponse.model_validate(user_message),
            ai_message=MessageDetailResponse.model_validate(ai_message),
            analysis=analyze_res,
            decision=decision_detail,
            ai_response_log=AIResponseDetailResponse.model_validate(ai_response_log)
        )


message_pipeline = MessagePipeline()
