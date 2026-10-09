import uuid
from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel, Field


class CreateThoughtRecordRequest(BaseModel):
    user_id: uuid.UUID
    conversation_id: Optional[uuid.UUID] = None
    situation: str = Field(..., min_length=3, description="What event or situation triggered the thought?")
    automatic_thought: str = Field(..., min_length=3, description="The negative automatic thought that occurred")
    initial_belief_rating: int = Field(80, ge=0, le=100, description="Initial conviction in this thought (0-100%)")
    emotions: Dict[str, int] = Field(default_factory=dict, description="Emotions and intensity before reframe")
    distortion_type: str = Field(..., description="Cognitive distortion identified by BERT or user")
    evidence_for: str = Field(..., description="Factual evidence supporting the automatic thought")
    evidence_against: str = Field(..., description="Factual evidence contradicting the automatic thought")
    balanced_thought: str = Field(..., min_length=3, description="Realistic, balanced alternative perspective")
    outcome_belief_rating: int = Field(20, ge=0, le=100, description="Conviction in automatic thought after reframe")
    outcome_emotions: Dict[str, int] = Field(default_factory=dict, description="Emotions and intensity after reframe")
    behavioral_action: Optional[str] = Field(None, description="Small behavioral experiment or action step")


class ThoughtRecordResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    conversation_id: Optional[uuid.UUID] = None
    situation: str
    automatic_thought: str
    initial_belief_rating: int
    emotions: Dict[str, int]
    distortion_type: str
    evidence_for: str
    evidence_against: str
    balanced_thought: str
    outcome_belief_rating: int
    outcome_emotions: Dict[str, int]
    behavioral_action: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
