import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories.user_repository import UserRepository
from app.db.models import EpisodicTherapyMemory
from app.schemas.auth import IntakeAssessmentRequest, IntakeAssessmentResponse
from app.schemas.response import PatientClinicalProfileResponse, EpisodicTherapyMemoryResponse
from app.services.clinical_memory_service import clinical_memory_service

router = APIRouter(prefix="/users", tags=["Intake & Clinical Memory"])


@router.post("/{user_id}/intake", response_model=IntakeAssessmentResponse, status_code=status.HTTP_201_CREATED)
def submit_intake_assessment(
    user_id: uuid.UUID,
    payload: IntakeAssessmentRequest,
    db: Session = Depends(get_db)
):
    user_repo = UserRepository(db)
    user = user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    assessment = user_repo.save_intake_assessment(user_id, payload)
    # Also initialize or sync clinical profile
    clinical_memory_service.get_or_create_profile(user_id, db)
    return assessment


@router.get("/{user_id}/intake", response_model=IntakeAssessmentResponse)
def get_intake_assessment(
    user_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    user_repo = UserRepository(db)
    assessment = user_repo.get_intake_assessment(user_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Intake assessment not found for this user.")

    return assessment


@router.get("/{user_id}/clinical-profile", response_model=PatientClinicalProfileResponse)
def get_clinical_profile(
    user_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    user_repo = UserRepository(db)
    user = user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    profile = clinical_memory_service.get_or_create_profile(user_id, db)
    return profile


@router.get("/{user_id}/memories", response_model=List[EpisodicTherapyMemoryResponse])
def get_episodic_memories(
    user_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    user_repo = UserRepository(db)
    user = user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    memories = (
        db.query(EpisodicTherapyMemory)
        .filter(EpisodicTherapyMemory.user_id == user_id)
        .order_by(EpisodicTherapyMemory.created_at.desc())
        .all()
    )
    return memories
