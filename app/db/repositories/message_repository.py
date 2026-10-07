import uuid
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.models import Message, AIResponse, Conversation


class MessageRepository:
    def __init__(self, db: Session):
        self.db = db

    def add_message(
        self,
        conversation_id: uuid.UUID,
        sender_type: str,
        content: str,
        commit: bool = True
    ) -> Message:
        # Lock conversation row using SELECT FOR UPDATE to prevent concurrent sequence race conditions
        self.db.query(Conversation).filter(Conversation.id == conversation_id).with_for_update().first()

        # Monotonic sequence counter per conversation
        max_seq = (
            self.db.query(func.max(Message.sequence_number))
            .filter(Message.conversation_id == conversation_id)
            .scalar()
        )
        next_seq = (max_seq or 0) + 1

        message = Message(
            conversation_id=conversation_id,
            sender_type=sender_type,
            content=content,
            sequence_number=next_seq
        )
        self.db.add(message)
        if commit:
            self.db.commit()
            self.db.refresh(message)
        else:
            self.db.flush()
        return message

    def add_ai_response(
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

    def get_message_by_id(self, message_id: uuid.UUID) -> Optional[Message]:
        return self.db.query(Message).filter(Message.id == message_id).first()

    def get_conversation_messages(self, conversation_id: uuid.UUID) -> List[Message]:
        return (
            self.db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.sequence_number.asc())
            .all()
        )
