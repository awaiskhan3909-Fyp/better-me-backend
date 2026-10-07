import uuid
from typing import Optional
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="User input text to analyze for cognitive distortions and safety risk", min_length=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "I feel like I ruin everything and I am a complete failure."
            }
        }
    }


class CreateConversationRequest(BaseModel):
    title: Optional[str] = Field(default="New Session", max_length=255)
    user_id: Optional[uuid.UUID] = Field(default=None, description="Optional UUID of existing user. If omitted, default demo user is assigned.")


class CreateMessageRequest(BaseModel):
    text: str = Field(..., description="User text message to record and analyze", min_length=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "I feel like I ruin everything and I am a complete failure."
            }
        }
    }
