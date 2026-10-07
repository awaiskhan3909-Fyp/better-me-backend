import uuid
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import CONVERSATION_HISTORY_LIMIT
from app.db.repositories.message_repository import MessageRepository
from app.db.models import Message


class HistoryMessageItem(BaseModel):
    id: uuid.UUID
    sender_type: str
    content: str
    sequence_number: int


class ConversationContext(BaseModel):
    conversation_id: uuid.UUID
    history_count: int
    has_history: bool
    recent_messages: List[HistoryMessageItem]
    last_user_text: Optional[str] = None
    last_ai_text: Optional[str] = None
    is_new_topic: bool = False


class ConversationManager:
    """
    Service responsible for retrieving and structuring conversation-level context.
    Decoupled from AI response generation and natural language synthesis.
    """
    def __init__(self, history_limit: int = CONVERSATION_HISTORY_LIMIT):
        self.history_limit = history_limit

    def get_conversation_context(
        self,
        conversation_id: uuid.UUID,
        db: Session,
        current_text: Optional[str] = None
    ) -> ConversationContext:
        msg_repo = MessageRepository(db)
        all_messages: List[Message] = msg_repo.get_conversation_messages(conversation_id)

        # Retrieve up to self.history_limit recent messages
        recent_raw = all_messages[-self.history_limit:] if all_messages else []

        recent_items = [
            HistoryMessageItem(
                id=m.id,
                sender_type=m.sender_type,
                content=m.content,
                sequence_number=m.sequence_number
            )
            for m in recent_raw
        ]

        last_user = next((m.content for m in reversed(recent_raw) if m.sender_type == "user"), None)
        last_ai = next((m.content for m in reversed(recent_raw) if m.sender_type == "ai"), None)

        is_new_topic = len(recent_raw) == 0

        return ConversationContext(
            conversation_id=conversation_id,
            history_count=len(all_messages),
            has_history=len(recent_raw) > 0,
            recent_messages=recent_items,
            last_user_text=last_user,
            last_ai_text=last_ai,
            is_new_topic=is_new_topic
        )


conversation_manager = ConversationManager()
