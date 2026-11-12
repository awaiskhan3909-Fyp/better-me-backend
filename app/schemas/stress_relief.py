import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class StressReliefRecordCreate(BaseModel):
    user_id: uuid.UUID = Field(..., description="ID of the user who completed the activity")
    activity_name: str = Field(..., description="Name of the activity, e.g., 4-7-8 Breathing Pacer")
    activity_type: str = Field(..., description="Type of activity: breathing, grounding, pmr, mindful_pause")
    duration_seconds: int = Field(default=0, description="Duration spent in seconds")
    notes: Optional[str] = Field(default=None, description="Optional brief notes or reflection")


class StressReliefRecordResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    activity_name: str
    activity_type: str
    duration_seconds: int
    is_completed: bool
    notes: Optional[str] = None
    completed_at: datetime
    created_at: datetime

    model_config = {"from_attributes": True}

