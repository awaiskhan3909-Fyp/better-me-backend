import uuid
from typing import List
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import StressReliefActivity, User
from app.schemas.stress_relief import StressReliefRecordCreate, StressReliefRecordResponse

router = APIRouter(prefix="/stress-relief", tags=["Instant Stress Relief"])


@router.post("/record", response_model=StressReliefRecordResponse, status_code=status.HTTP_201_CREATED)
def record_stress_relief_activity(
    payload: StressReliefRecordCreate,
    db: Session = Depends(get_db)
):
    """Records the completion of a self-help instant stress relief exercise."""
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    record = StressReliefActivity(
        user_id=payload.user_id,
        activity_name=payload.activity_name,
        activity_type=payload.activity_type,
        duration_seconds=payload.duration_seconds,
        is_completed=True,
        notes=payload.notes,
        completed_at=datetime.now(timezone.utc)
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.get("/history", response_model=List[StressReliefRecordResponse])
def get_stress_relief_history(
    user_id: uuid.UUID = Query(..., description="ID of the user"),
    limit: int = Query(default=10, ge=1, le=50, description="Max activities to return"),
    db: Session = Depends(get_db)
):
    """Retrieves recent stress relief activities for the user."""
    records = (
        db.query(StressReliefActivity)
        .filter(StressReliefActivity.user_id == user_id)
        .order_by(StressReliefActivity.completed_at.desc())
        .limit(limit)
        .all()
    )
    return records

