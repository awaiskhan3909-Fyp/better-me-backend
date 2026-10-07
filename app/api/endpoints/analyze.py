from fastapi import APIRouter, HTTPException
from app.schemas.request import AnalyzeRequest
from app.schemas.response import AnalyzeResponse, CBTGuidance, DistortionPrediction
from app.services.safety_service import safety_service
from app.services.distortion_service import distortion_service
from app.services.ner_service import ner_service
from app.services.cbt_service import cbt_service
from app.services.response_decision_service import (
    response_decision_service,
    DecisionContext,
    ResponseStrategy,
)
from app.services.llm_service import llm_service, LLMResult

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(payload: AnalyzeRequest):
    """
    Stateless Analysis Endpoint:
    1. Validates input text.
    2. Runs Safety, Distortion, and NER prediction models.
    3. Evaluates Response Strategy via ResponseDecisionService.
    4. Routes through LLM Service (grounded by Response Strategy).
    5. Returns decision-guided response payload.
    """
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # 1. Run safety, distortion, and NER prediction models
    safety_pred = safety_service.predict(text)
    distortion_pred = distortion_service.predict(text)
    entities = ner_service.extract_entities(text)

    # 2. Evaluate Response Strategy via Response Decision Engine
    decision_context = DecisionContext(
        user_text=text,
        safety_result=safety_pred,
        distortion_result=distortion_pred,
        entities=entities,
        conversation_context=None
    )
    decision = response_decision_service.evaluate_decision(decision_context)

    # 3. Route to LLM Generation Layer
    default_cbt_guidance = None
    if decision.strategy == ResponseStrategy.CBT_SUPPORT:
        default_cbt_guidance = cbt_service.get_cbt_guidance(text, distortion_pred.predicted_class)

    llm_result: LLMResult = llm_service.generate_response(
        user_text=text,
        strategy=decision.strategy.value,
        decision_reason=decision.reason,
        safety_risk_level=safety_pred.risk_level,
        distortion_pred=distortion_pred,
        entities=entities,
        cbt_guidance=default_cbt_guidance,
        conversation_context=None
    )

    ai_message_content = llm_result.content

    # Configure response schemas
    if decision.strategy == ResponseStrategy.CBT_SUPPORT and default_cbt_guidance:
        cbt_guidance = default_cbt_guidance
        cbt_guidance.balanced_thought_guidance = ai_message_content
        distortion_output = distortion_pred
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
            confidence=0.0,
            all_probabilities=distortion_pred.all_probabilities
        )
    else:
        cbt_guidance = CBTGuidance(
            detected_distortion="None",
            template_id=None,
            title=None,
            explanation=None,
            reframing_question=None,
            balanced_thought_guidance=ai_message_content,
            small_action=None
        )
        distortion_output = DistortionPrediction(
            predicted_class="None",
            confidence=0.0,
            all_probabilities=distortion_pred.all_probabilities
        )

    return AnalyzeResponse(
        text=text,
        safety=safety_pred,
        distortion=distortion_output,
        entities=entities,
        cbt_guidance=cbt_guidance
    )
