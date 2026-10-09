import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories.user_repository import UserRepository
from app.schemas.auth import IntakeAssessmentRequest, IntakeAssessmentResponse

router = APIRouter(prefix="/users", tags=["Intake Assessment"])


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
