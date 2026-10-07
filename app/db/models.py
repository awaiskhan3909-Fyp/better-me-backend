import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import (
    Column, String, Text, Boolean, Integer, Float, DateTime, ForeignKey,
    UniqueConstraint, Index, JSON
)
from sqlalchemy.types import UUID
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.db.database import Base

# Cross-database JSON abstraction (Renders JSON on SQLite, JSONB on PostgreSQL)
JSON_TYPE = JSON().with_variant(JSONB, "postgresql")


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    conversations: Mapped[List["Conversation"]] = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), default="New Session", nullable=False)
    current_risk_level: Mapped[str] = mapped_column(String(20), default="Safe", nullable=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.sequence_number")
    analyses: Mapped[List["MessageAnalysis"]] = relationship("MessageAnalysis", back_populates="conversation", cascade="all, delete-orphan")
    ai_responses: Mapped[List["AIResponse"]] = relationship("AIResponse", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    sender_type: Mapped[str] = mapped_column(String(20), nullable=False)  # 'user', 'ai', 'system'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint("conversation_id", "sequence_number", name="uq_messages_conversation_sequence"),
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")
    analysis: Mapped[Optional["MessageAnalysis"]] = relationship("MessageAnalysis", back_populates="message", uselist=False, cascade="all, delete-orphan")
    prompt_responses: Mapped[List["AIResponse"]] = relationship("AIResponse", foreign_keys="[AIResponse.user_message_id]", back_populates="user_message")
    generated_ai_response: Mapped[Optional["AIResponse"]] = relationship("AIResponse", foreign_keys="[AIResponse.ai_message_id]", back_populates="ai_message", uselist=False)


class MessageAnalysis(Base):
    __tablename__ = "message_analyses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)

    # Safety Model Outputs
    safety_risk_level: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    needs_safety_alert: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    safety_probabilities: Mapped[dict] = mapped_column(JSON_TYPE, nullable=False)

    # Distortion Model Outputs
    distortion_class: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    distortion_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    distortion_probabilities: Mapped[dict] = mapped_column(JSON_TYPE, nullable=False)

    # NER Entities Output
    entities: Mapped[list] = mapped_column(JSON_TYPE, default=list, nullable=False)

    # Model Metadata
    model_metadata: Mapped[dict] = mapped_column(JSON_TYPE, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    message: Mapped["Message"] = relationship("Message", back_populates="analysis")
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="analyses")


class AIResponse(Base):
    __tablename__ = "ai_responses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    user_message_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True, index=True)
    ai_message_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=True, index=True)

    response_source: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # 'cbt_template_engine', 'conversational_llm', 'safety_override_engine'
    response_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)    # 'therapeutic_reframe', 'crisis_intervention', 'conversational_dialogue'
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    response_content: Mapped[str] = mapped_column(Text, nullable=False)

    cbt_data: Mapped[Optional[dict]] = mapped_column(JSON_TYPE, nullable=True)
    llm_metadata: Mapped[Optional[dict]] = mapped_column(JSON_TYPE, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column("metadata", JSON_TYPE, default=dict, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="ai_responses")
    user_message: Mapped[Optional["Message"]] = relationship("Message", foreign_keys=[user_message_id], back_populates="prompt_responses")
    ai_message: Mapped[Optional["Message"]] = relationship("Message", foreign_keys=[ai_message_id], back_populates="generated_ai_response")
