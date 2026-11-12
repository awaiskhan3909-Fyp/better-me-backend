import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class CBTExerciseSubmitRequest(BaseModel):
    user_id: uuid.UUID
    distortion_category: str
    exercise_id: str
    exercise_title: str
    scenario: str
    answers: Dict[str, str] = Field(default_factory=dict)
    is_completed: bool = True


class CBTExerciseResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    distortion_category: str
    exercise_id: str
    exercise_title: str
    scenario: str
    answers: Dict[str, str]
    is_completed: bool
    completed_at: datetime
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CBTExerciseCategoryProgress(BaseModel):
    completed: int
    total: int = 4
    percentage: float


class CBTExerciseProgressSummary(BaseModel):
    total_completed: int
    total_available: int = 24
    completion_percentage: float
    categories_progress: Dict[str, CBTExerciseCategoryProgress]
    completed_exercise_ids: List[str]
    recent_activity: List[CBTExerciseResponse]

