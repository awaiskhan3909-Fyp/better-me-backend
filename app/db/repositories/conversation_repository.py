import uuid
from typing import List, Optional
from sqlalchemy.orm import Session
from app.db.models import Conversation


class ConversationRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_conversation(self, user_id: uuid.UUID, title: str = "New Session", commit: bool = True) -> Conversation:
        conversation = Conversation(
            user_id=user_id,
            title=title
        )
        self.db.add(conversation)
        if commit:
            self.db.commit()
            self.db.refresh(conversation)
        else:
            self.db.flush()
        return conversation

    def get_by_id(self, conversation_id: uuid.UUID) -> Optional[Conversation]:
        return self.db.query(Conversation).filter(Conversation.id == conversation_id).first()

    def get_user_conversations(self, user_id: uuid.UUID) -> List[Conversation]:
        return (
            self.db.query(Conversation)
            .filter(Conversation.user_id == user_id, Conversation.is_archived == False)
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    def update_risk_level(self, conversation_id: uuid.UUID, risk_level: str, commit: bool = True) -> Optional[Conversation]:
        conversation = self.get_by_id(conversation_id)
        if conversation:
            conversation.current_risk_level = risk_level
            if commit:
                self.db.commit()
                self.db.refresh(conversation)
            else:
                self.db.flush()
        return conversation
