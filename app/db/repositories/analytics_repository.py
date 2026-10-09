import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.models import User, Conversation, Message, MessageAnalysis, UserIntakeAssessment
from app.schemas.auth import (
    DashboardStatsResponse,
    SessionSummaryItem,
    EmotionalTrendItem
)


class AnalyticsRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_dashboard_stats(self, user_id: uuid.UUID) -> DashboardStatsResponse:
        user = self.db.query(User).filter(User.id == user_id).first()
        user_name = user.full_name or user.email.split("@")[0] if user else "Friend"

        intake = self.db.query(UserIntakeAssessment).filter(UserIntakeAssessment.user_id == user_id).first()
        primary_focus = intake.primary_focus if intake else []
        primary_goal = intake.primary_goal if intake else "Cognitive Reframing & Emotional Wellbeing"

        # Conversations
        conversations = (
            self.db.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.created_at.desc())
            .all()
        )
        total_sessions = len(conversations)

        # Messages & Analyses
        conv_ids = [c.id for c in conversations]

        total_messages = 0
        distortions_breakdown: Dict[str, int] = {
            "Catastrophizing": 0,
            "Mind Reading": 0,
            "Overgeneralization": 0,
            "All-or-Nothing Thinking": 0,
            "Emotional Reasoning": 0,
            "Fortune Telling": 0
        }

        recent_sessions: List[SessionSummaryItem] = []
        emotional_trends: List[EmotionalTrendItem] = []

        if conv_ids:
            total_messages = (
                self.db.query(Message)
                .filter(Message.conversation_id.in_(conv_ids))
                .count()
            )

            # Get distortion frequencies across all user messages
            analyses = (
                self.db.query(MessageAnalysis)
                .filter(MessageAnalysis.conversation_id.in_(conv_ids))
                .all()
            )

            for a in analyses:
                if a.distortion_class and a.distortion_class in distortions_breakdown:
                    distortions_breakdown[a.distortion_class] += 1

            # Format recent sessions
            for conv in conversations[:6]:
                # Count messages in this session
                msg_count = len(conv.messages) if conv.messages else 0
                
                # Distortions detected in this session
                session_distortions = list({
                    an.distortion_class
                    for an in conv.analyses
                    if an.distortion_class and an.distortion_class not in ["None", "Normal", "no distortion"]
                })

                # Calculate duration approximation (based on message count, min 5 mins)
                approx_duration = max(5, round(msg_count * 2.5))

                recent_sessions.append(
                    SessionSummaryItem(
                        id=str(conv.id),
                        date=conv.created_at.strftime("%Y-%m-%d"),
                        title=conv.title or "Therapy Session",
                        duration=approx_duration,
                        distortions_detected=session_distortions,
                        risk_level=conv.current_risk_level.lower() if conv.current_risk_level else "low",
                        message_count=msg_count
                    )
                )

                # Daily emotional trend item
                daily_counts: Dict[str, int] = {k: 0 for k in distortions_breakdown.keys()}
                for an in conv.analyses:
                    if an.distortion_class in daily_counts:
                        daily_counts[an.distortion_class] += 1

                emotional_trends.append(
                    EmotionalTrendItem(
                        date=conv.created_at.strftime("%Y-%m-%d"),
                        distortion_counts=daily_counts,
                        total_messages=msg_count,
                        risk_level=conv.current_risk_level.lower() if conv.current_risk_level else "low"
                    )
                )

        avg_duration = round(sum(s.duration for s in recent_sessions) / len(recent_sessions)) if recent_sessions else 0
        current_risk = conversations[0].current_risk_level if conversations else "Safe"

        return DashboardStatsResponse(
            user_name=user_name,
            total_sessions=total_sessions,
            avg_duration_minutes=avg_duration,
            total_messages=total_messages,
            current_risk_level=current_risk,
            primary_focus=primary_focus,
            primary_goal=primary_goal,
            distortions_breakdown=distortions_breakdown,
            recent_sessions=recent_sessions,
            emotional_trends=emotional_trends
        )
