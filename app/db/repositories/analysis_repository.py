import uuid
from typing import Optional, List
from sqlalchemy.orm import Session
from app.db.models import MessageAnalysis


class AnalysisRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_analysis(
        self,
        message_id: uuid.UUID,
        conversation_id: uuid.UUID,
        safety_risk_level: str,
        needs_safety_alert: bool,
        safety_probabilities: dict,
        distortion_class: str,
        distortion_confidence: float,
        distortion_probabilities: dict,
        entities: list,
        model_metadata: Optional[dict] = None,
        commit: bool = True,
    ) -> MessageAnalysis:
        analysis = MessageAnalysis(
            message_id=message_id,
            conversation_id=conversation_id,
            safety_risk_level=safety_risk_level,
            needs_safety_alert=needs_safety_alert,
            safety_probabilities=safety_probabilities,
            distortion_class=distortion_class,
            distortion_confidence=distortion_confidence,
            distortion_probabilities=distortion_probabilities,
            entities=entities,
            model_metadata=model_metadata or {},
        )
        self.db.add(analysis)
        if commit:
            self.db.commit()
            self.db.refresh(analysis)
        else:
            self.db.flush()
        return analysis

    def get_by_message_id(self, message_id: uuid.UUID) -> Optional[MessageAnalysis]:
        return self.db.query(MessageAnalysis).filter(MessageAnalysis.message_id == message_id).first()

    def get_conversation_analyses(self, conversation_id: uuid.UUID) -> List[MessageAnalysis]:
        return (
            self.db.query(MessageAnalysis)
            .filter(MessageAnalysis.conversation_id == conversation_id)
            .order_by(MessageAnalysis.created_at.asc())
            .all()
        )
