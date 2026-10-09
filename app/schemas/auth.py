import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, EmailStr, Field


# --- Authentication Schemas ---

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, description="Password with minimum 6 characters")
    full_name: Optional[str] = Field(default=None, max_length=100)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    full_name: Optional[str] = None
    is_active: bool = True
    has_completed_intake: bool = False
    created_at: datetime

    model_config = {"from_attributes": True}


class AuthResponse(BaseModel):
    user: UserResponse
    message: str = "Success"


# --- Clinical Intake Assessment Schemas ---

class IntakeAssessmentRequest(BaseModel):
    primary_focus: List[str] = Field(default_factory=list, description="Primary focus areas (e.g. Academic Stress, Anxiety)")
    distress_baseline: int = Field(default=0, ge=0, le=12, description="Distress rating baseline score (0-12)")
    familiar_distortions: List[str] = Field(default_factory=list, description="Pre-screened distorted thinking patterns")
    primary_goal: str = Field(default="Reframing negative thoughts", max_length=255)
    safety_acknowledged: bool = Field(default=True, description="Confirmation of emergency safety helpline understanding")


class IntakeAssessmentResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    primary_focus: List[str]
    distress_baseline: int
    familiar_distortions: List[str]
    primary_goal: str
    safety_acknowledged: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Dynamic Dashboard & Analytics Schemas ---

class SessionSummaryItem(BaseModel):
    id: str
    date: str
    title: str
    duration: int  # in minutes
    distortions_detected: List[str]
    risk_level: str
    message_count: int


class EmotionalTrendItem(BaseModel):
    date: str
    distortion_counts: Dict[str, int]
    total_messages: int
    risk_level: str


class DashboardStatsResponse(BaseModel):
    user_name: str
    total_sessions: int
    avg_duration_minutes: int
    total_messages: int
    current_risk_level: str
    primary_focus: List[str]
    primary_goal: Optional[str] = None
    distortions_breakdown: Dict[str, int]
    recent_sessions: List[SessionSummaryItem]
    emotional_trends: List[EmotionalTrendItem]
