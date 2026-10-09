import uuid
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import (
    CBTThoughtRecord,
    User,
    PatientClinicalProfile,
    EpisodicTherapyMemory
)
from app.schemas.thought_record import (
    CreateThoughtRecordRequest,
    ThoughtRecordResponse
)
from app.services.clinical_memory_service import clinical_memory_service

router = APIRouter(prefix="/thought-records", tags=["CBT Thought Records"])


@router.post("", response_model=ThoughtRecordResponse, status_code=status.HTTP_201_CREATED)
def create_thought_record(
    payload: CreateThoughtRecordRequest,
    db: Session = Depends(get_db)
):
    """
    Submits a completed Beckian 5-Column Thought Record.
    Automatically closes the clinical loop by:
    1. Storing the thought record.
    2. Logging an EpisodicTherapyMemory breakthrough.
    3. Updating PatientClinicalProfile (effective_reframes, active_homework, distortion frequency).
    """
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    record = CBTThoughtRecord(
        user_id=payload.user_id,
        conversation_id=payload.conversation_id,
        situation=payload.situation,
        automatic_thought=payload.automatic_thought,
        initial_belief_rating=payload.initial_belief_rating,
        emotions=payload.emotions,
        distortion_type=payload.distortion_type,
        evidence_for=payload.evidence_for,
        evidence_against=payload.evidence_against,
        balanced_thought=payload.balanced_thought,
        outcome_belief_rating=payload.outcome_belief_rating,
        outcome_emotions=payload.outcome_emotions,
        behavioral_action=payload.behavioral_action,
    )
    db.add(record)

    # 1. Add to Episodic Therapy Memories for future LLM context recall
    breakthrough_summary = (
        f"Belief dropped from {payload.initial_belief_rating}% to {payload.outcome_belief_rating}%. "
        f"Action plan: {payload.behavioral_action or 'Reflective practice'}."
    )
    episode = EpisodicTherapyMemory(
        user_id=payload.user_id,
        conversation_id=payload.conversation_id,
        situation_context=payload.situation[:180],
        distorted_thought=payload.automatic_thought[:200],
        distortion_type=payload.distortion_type,
        rational_reframe=payload.balanced_thought[:250],
        breakthrough_notes=breakthrough_summary[:250],
    )
    db.add(episode)

    # 2. Update Patient Clinical Profile
    profile = clinical_memory_service.get_or_create_profile(payload.user_id, db)

    # Record reframe
    reframes = list(profile.effective_reframes or [])
    if payload.balanced_thought not in reframes:
        reframes.append(payload.balanced_thought[:200])
        profile.effective_reframes = reframes[-10:]  # Keep last 10 effective reframes

    # Update active homework if provided
    if payload.behavioral_action:
        profile.active_homework = payload.behavioral_action

    # Update dominant distortions
    if payload.distortion_type and payload.distortion_type != "None":
        dist_map = dict(profile.dominant_distortions or {})
        dist_map[payload.distortion_type] = dist_map.get(payload.distortion_type, 0) + 1
        profile.dominant_distortions = dist_map

    profile.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(record)
    return record


@router.get("", response_model=List[ThoughtRecordResponse])
def get_thought_records(
    user_id: uuid.UUID = Query(..., description="ID of the user"),
    db: Session = Depends(get_db)
):
    """Retrieves all completed thought records for a user, newest first."""
    records = (
        db.query(CBTThoughtRecord)
        .filter(CBTThoughtRecord.user_id == user_id)
        .order_by(CBTThoughtRecord.created_at.desc())
        .all()
    )
    return records


@router.get("/{record_id}", response_model=ThoughtRecordResponse)
def get_thought_record_by_id(
    record_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Retrieves a single thought record by its ID."""
    record = db.query(CBTThoughtRecord).filter(CBTThoughtRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Thought record not found.")
    return record


@router.delete("/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_thought_record(
    record_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Deletes a thought record."""
    record = db.query(CBTThoughtRecord).filter(CBTThoughtRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Thought record not found.")
    db.delete(record)
    db.commit()
    return None
