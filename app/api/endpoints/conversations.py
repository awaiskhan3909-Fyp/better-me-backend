import uuid
import re
import logging
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.db.database import get_db
from app.db.models import EpisodicTherapyMemory
from app.db.repositories.user_repository import UserRepository
from app.db.repositories.conversation_repository import ConversationRepository
from app.db.repositories.message_repository import MessageRepository
from app.db.repositories.analysis_repository import AnalysisRepository
from app.db.repositories.ai_response_repository import AIResponseRepository

from app.schemas.request import CreateConversationRequest, CreateMessageRequest, UpdateConversationRequest
from app.schemas.response import (
    ConversationResponse,
    MessageDetailResponse,
    ConversationMessageResponse,
    AnalyzeResponse,
    AIResponseDetailResponse,
    ResponseDecisionDetail,
    CBTGuidance,
    DistortionPrediction,
)

from app.services.clinical_memory_service import clinical_memory_service
from app.services.session_state_service import session_state_service, TherapyStage
from app.services.pdf_report_service import pdf_report_service
from app.services.message_pipeline import message_pipeline

logger = logging.getLogger("better_me.conversations")

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.get("", response_model=List[ConversationResponse])
def list_user_conversations(
    user_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Retrieves all past conversations for a user, ordered by most recent.
    Strictly isolates per-account conversations (never leaks demo user's chats
    when a specific user_id is supplied).
    """
    user_repo = UserRepository(db)
    conv_repo = ConversationRepository(db)

    if user_id:
        try:
            parsed_uuid = uuid.UUID(str(user_id))
        except (ValueError, TypeError):
            return []
        user = user_repo.get_by_id(parsed_uuid)
        if not user:
            return []
        return conv_repo.get_user_conversations(user.id)

    demo_user = user_repo.get_or_create_default_user()
    return conv_repo.get_user_conversations(demo_user.id)


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: CreateConversationRequest = CreateConversationRequest(),
    db: Session = Depends(get_db)
):
    """
    Creates a new conversation session.
    If user_id is omitted or invalid, defaults to the default system user.
    """
    user_repo = UserRepository(db)
    conv_repo = ConversationRepository(db)

    target_user = None
    if payload.user_id:
        try:
            target_user = user_repo.get_by_id(payload.user_id)
        except Exception:
            pass

    if not target_user:
        target_user = user_repo.get_or_create_default_user()

    title = payload.title.strip() if payload.title else "New Session"
    conversation = conv_repo.create_conversation(user_id=target_user.id, title=title)
    return conversation


@router.get("/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Retrieves conversation metadata by conversation ID."""
    conv_repo = ConversationRepository(db)
    conversation = conv_repo.get_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


@router.patch("/{conversation_id}", response_model=ConversationResponse)
def update_conversation_title(
    conversation_id: uuid.UUID,
    payload: UpdateConversationRequest,
    db: Session = Depends(get_db)
):
    """Updates conversation title (renaming)."""
    conv_repo = ConversationRepository(db)
    conversation = conv_repo.update_title(conversation_id, payload.title)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


@router.delete("/{conversation_id}", status_code=status.HTTP_200_OK)
def delete_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Deletes a conversation and all its messages."""
    conv_repo = ConversationRepository(db)
    success = conv_repo.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"status": "success", "message": "Conversation deleted successfully."}


@router.get("/{conversation_id}/messages", response_model=List[MessageDetailResponse])
def get_conversation_messages(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Retrieves all messages for a given conversation ordered by sequence number ascending."""
    conv_repo = ConversationRepository(db)
    msg_repo = MessageRepository(db)

    conversation = conv_repo.get_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages = msg_repo.get_conversation_messages(conversation_id)
    return messages


@router.post("/{conversation_id}/messages", response_model=ConversationMessageResponse, status_code=status.HTTP_201_CREATED)
def send_message_in_conversation(
    conversation_id: uuid.UUID,
    payload: CreateMessageRequest,
    db: Session = Depends(get_db)
):
    """
    Stateful Message Processing Endpoint:
    Validates conversation_id and input text, invokes the MessagePipeline,
    and returns the structured clinical CBT message response.
    """
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message text cannot be empty.")

    conv_repo = ConversationRepository(db)
    conversation = conv_repo.get_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return message_pipeline.process(
        conversation_id=conversation_id,
        text=text,
        db=db,
        conversation=conversation
    )


@router.get("/{conversation_id}/report", summary="Download professional CBT PDF session report")
def download_conversation_report(
    conversation_id: uuid.UUID,
    user_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Generates and streams a professional, Beckian-compliant psychological session report in PDF format.
    Ensures authorized patient data isolation.
    """
    conv_repo = ConversationRepository(db)
    conversation = conv_repo.get_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    user_repo = UserRepository(db)
    demo_user = user_repo.get_or_create_default_user()
    if user_id:
        try:
            req_user_uuid = uuid.UUID(str(user_id))
            if conversation.user_id != req_user_uuid and conversation.user_id != demo_user.id:
                raise HTTPException(status_code=403, detail="Unauthorized access to this session report.")
            if conversation.user_id == demo_user.id and req_user_uuid != demo_user.id:
                conversation.user_id = req_user_uuid
                db.commit()
        except ValueError:
            pass

    try:
        pdf_buffer = pdf_report_service.generate_pdf_buffer(conversation, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate session PDF report: {str(e)}")

    date_str = conversation.created_at.strftime("%Y%m%d")
    filename = f"BetterMe_Session_Report_{date_str}_{str(conversation_id)[:8]}.pdf"

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )


@router.get("/{conversation_id}/report/summary", summary="Retrieve session report summary data in JSON format")
def get_conversation_report_summary(
    conversation_id: uuid.UUID,
    user_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Retrieves the parsed psychological session synthesis data in JSON format for frontend rendering.
    """
    conv_repo = ConversationRepository(db)
    conversation = conv_repo.get_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    user_repo = UserRepository(db)
    demo_user = user_repo.get_or_create_default_user()
    if user_id:
        try:
            req_user_uuid = uuid.UUID(str(user_id))
            if conversation.user_id != req_user_uuid and conversation.user_id != demo_user.id:
                raise HTTPException(status_code=403, detail="Unauthorized access to this session report.")
            if conversation.user_id == demo_user.id and req_user_uuid != demo_user.id:
                conversation.user_id = req_user_uuid
                db.commit()
        except ValueError:
            pass

    try:
        report_data = pdf_report_service.extract_session_data(conversation, db)
        return report_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract session summary: {str(e)}")


@router.post("/{conversation_id}/complete", summary="Complete and finalize a CBT session")
def complete_conversation_session(
    conversation_id: uuid.UUID,
    user_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Formally completes a CBT therapy session:
    1. Sets session state current_stage to 'closure' and status to 'completed' with timestamp.
    2. Extracts real session summary (topic, distortions, user evidence lines, mood, homework, next focus).
    3. Persists tailored homework and rational reframe into session state.
    4. Records an EpisodicTherapyMemory entry for longitudinal clinical continuity.
    5. Returns finalized JSON report summary.
    """
    conv_repo = ConversationRepository(db)
    conversation = conv_repo.get_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    user_repo = UserRepository(db)
    demo_user = user_repo.get_or_create_default_user()
    if user_id:
        try:
            req_user_uuid = uuid.UUID(str(user_id))
            if conversation.user_id != req_user_uuid and conversation.user_id != demo_user.id:
                raise HTTPException(status_code=403, detail="Unauthorized access to this session.")
            if conversation.user_id == demo_user.id and req_user_uuid != demo_user.id:
                conversation.user_id = req_user_uuid
                db.commit()
        except ValueError:
            pass

    # Ensure session state exists and mark as completed
    session_state = session_state_service.get_or_create_session_state(conversation.id, db)
    session_state.current_stage = TherapyStage.CLOSURE.value

    meta = dict(session_state.metadata_json or {})
    meta["status"] = "completed"
    meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    session_state.metadata_json = meta
    flag_modified(session_state, "metadata_json")

    # Extract verified session data
    report_data = pdf_report_service.extract_session_data(conversation, db)

    # Persist tailored homework in session_state if not already present
    if not session_state.action_plan and report_data.get("homework"):
        session_state.action_plan = report_data.get("homework")
    if not session_state.balanced_perspective and report_data.get("balanced_thought"):
        session_state.balanced_perspective = report_data.get("balanced_thought")
    if not session_state.identified_trigger and report_data.get("presenting_concern"):
        session_state.identified_trigger = report_data.get("presenting_concern")

    # Record longitudinal episodic therapy memory for past breakthrough continuity
    if len(conversation.messages or []) >= 2:
        try:
            existing_ep = db.query(EpisodicTherapyMemory).filter(
                EpisodicTherapyMemory.user_id == conversation.user_id,
                EpisodicTherapyMemory.situation_context == report_data.get("presenting_concern")
            ).first()
            if not existing_ep:
                new_ep = EpisodicTherapyMemory(
                    user_id=conversation.user_id,
                    conversation_id=conversation.id,
                    situation_context=report_data.get("presenting_concern") or "CBT Session",
                    distorted_thought=report_data.get("automatic_thought") or "Self-critical thought",
                    distortion_type=report_data.get("distortion_name") or "Cognitive Distortion",
                    rational_reframe=report_data.get("balanced_thought") or "Balanced alternative perspective",
                    breakthrough_notes=f"Reframed with CBT. Homework: {report_data.get('homework') or 'Thought monitoring'}"
                )
                db.add(new_ep)

            # Update patient longitudinal profile
            profile = clinical_memory_service.get_or_create_profile(conversation.user_id, db)
            profile.total_sessions_completed = (profile.total_sessions_completed or 0) + 1
            if report_data.get("homework"):
                profile.active_homework = report_data.get("homework")
            if report_data.get("presenting_concern"):
                profile.last_session_summary = report_data.get("presenting_concern")
            profile.updated_at = datetime.now(timezone.utc)
        except Exception:
            pass

    db.commit()
    db.refresh(conversation)

    return report_data


