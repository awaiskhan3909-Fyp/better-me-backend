import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class EntityItem(BaseModel):
    text: str
    label: str
    start: Optional[int] = None
    end: Optional[int] = None


class DistortionPrediction(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]


class SafetyPrediction(BaseModel):
    risk_level: str  # "Safe", "Moderate", "High Risk"
    needs_safety_alert: bool
    probabilities: Dict[str, float]


class CBTGuidance(BaseModel):
    detected_distortion: str
    template_id: Optional[int] = None
    title: Optional[str] = None
    explanation: Optional[str] = None
    reframing_question: Optional[str] = None
    balanced_thought_guidance: str
    small_action: Optional[str] = None


class AnalyzeResponse(BaseModel):
    text: str
    safety: SafetyPrediction
    distortion: DistortionPrediction
    entities: List[EntityItem]
    cbt_guidance: CBTGuidance


# --- Database & Intelligence Schemas ---

class ResponseDecisionDetail(BaseModel):
    strategy: str
    priority: str
    reason: str
    distortion_detected: Optional[str] = None
    safety_risk_level: str = "Safe"
    decision_version: str = "v1"


class ConversationResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    title: str
    current_risk_level: str
    is_archived: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MessageDetailResponse(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    sender_type: str
    content: str
    sequence_number: int
    created_at: datetime

    model_config = {"from_attributes": True}


class AnalysisDetailResponse(BaseModel):
    id: uuid.UUID
    message_id: uuid.UUID
    conversation_id: uuid.UUID
    safety_risk_level: str
    needs_safety_alert: bool
    distortion_class: str
    distortion_confidence: float
    entities: List[Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class AIResponseDetailResponse(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    user_message_id: Optional[uuid.UUID] = None
    ai_message_id: Optional[uuid.UUID] = None
    response_source: str
    response_type: str
    model_name: str
    response_content: str
    cbt_data: Optional[Dict[str, Any]] = None
    llm_metadata: Optional[Dict[str, Any]] = None
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationMessageResponse(BaseModel):
    conversation_id: uuid.UUID
    user_message: MessageDetailResponse
    ai_message: MessageDetailResponse
    analysis: AnalyzeResponse
    decision: ResponseDecisionDetail
    ai_response_log: AIResponseDetailResponse
