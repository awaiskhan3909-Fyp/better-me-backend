import uuid
from typing import Optional, List
from sqlalchemy.orm import Session
from app.db.models import AIResponse


class AIResponseRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_ai_response(
        self,
        conversation_id: uuid.UUID,
        response_source: str,
        response_type: str,
        model_name: str,
        response_content: str,
        user_message_id: Optional[uuid.UUID] = None,
        ai_message_id: Optional[uuid.UUID] = None,
        cbt_data: Optional[dict] = None,
        llm_metadata: Optional[dict] = None,
        metadata_json: Optional[dict] = None,
        commit: bool = True,
    ) -> AIResponse:
        ai_resp = AIResponse(
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            ai_message_id=ai_message_id,
            response_source=response_source,
            response_type=response_type,
            model_name=model_name,
            response_content=response_content,
            cbt_data=cbt_data,
            llm_metadata=llm_metadata,
            metadata_json=metadata_json or {},
        )
        self.db.add(ai_resp)
        if commit:
            self.db.commit()
            self.db.refresh(ai_resp)
        else:
            self.db.flush()
        return ai_resp

    def get_by_user_message_id(self, user_message_id: uuid.UUID) -> Optional[AIResponse]:
        return self.db.query(AIResponse).filter(AIResponse.user_message_id == user_message_id).first()

    def get_by_ai_message_id(self, ai_message_id: uuid.UUID) -> Optional[AIResponse]:
        return self.db.query(AIResponse).filter(AIResponse.ai_message_id == ai_message_id).first()

    def get_conversation_responses(self, conversation_id: uuid.UUID) -> List[AIResponse]:
        return (
            self.db.query(AIResponse)
            .filter(AIResponse.conversation_id == conversation_id)
            .order_by(AIResponse.created_at.asc())
            .all()
        )
