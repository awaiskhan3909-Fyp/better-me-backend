import uuid
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import (
    CBTExerciseSubmission,
    User,
    EpisodicTherapyMemory,
    PatientClinicalProfile
)
from app.schemas.cbt_exercise import (
    CBTExerciseSubmitRequest,
    CBTExerciseResponse,
    CBTExerciseProgressSummary,
    CBTExerciseCategoryProgress
)
from app.services.clinical_memory_service import clinical_memory_service

router = APIRouter(prefix="/cbt-exercises", tags=["CBT Practice Exercises"])

ALL_DISTORTION_CATEGORIES = [
    "Catastrophizing",
    "Mind Reading",
    "Overgeneralization",
    "All-or-Nothing Thinking",
    "Emotional Reasoning",
    "Fortune Telling"
]


@router.post("/submit", response_model=CBTExerciseResponse, status_code=status.HTTP_200_OK)
def submit_cbt_exercise(
    payload: CBTExerciseSubmitRequest,
    db: Session = Depends(get_db)
):
    """
    Submits or updates a guided practice exercise for a cognitive distortion.
    Prevents duplicate submissions for the same (user_id, exercise_id) by updating the existing entry.
    Also registers a clinical breakthrough in EpisodicTherapyMemory.
    """
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Check for existing submission to prevent duplicates and enable re-visitation
    existing = (
        db.query(CBTExerciseSubmission)
        .filter(
            CBTExerciseSubmission.user_id == payload.user_id,
            CBTExerciseSubmission.exercise_id == payload.exercise_id
        )
        .first()
    )

    now = datetime.now(timezone.utc)

    if existing:
        existing.distortion_category = payload.distortion_category
        existing.exercise_title = payload.exercise_title
        existing.scenario = payload.scenario
        existing.answers = payload.answers
        existing.is_completed = payload.is_completed
        existing.completed_at = now
        existing.updated_at = now
        record = existing
    else:
        record = CBTExerciseSubmission(
            user_id=payload.user_id,
            distortion_category=payload.distortion_category,
            exercise_id=payload.exercise_id,
            exercise_title=payload.exercise_title,
            scenario=payload.scenario,
            answers=payload.answers,
            is_completed=payload.is_completed,
            completed_at=now,
            created_at=now,
            updated_at=now
        )
        db.add(record)

    # Sync breakthrough with EpisodicTherapyMemory and Clinical Profile
    try:
        q1_thought = payload.answers.get("q1") or payload.exercise_title
        q5_reframe = payload.answers.get("q5") or payload.answers.get("q4") or "Balanced cognitive reframe"

        episode = EpisodicTherapyMemory(
            user_id=payload.user_id,
            situation_context=payload.scenario[:180],
            distorted_thought=q1_thought[:200],
            distortion_type=payload.distortion_category,
            rational_reframe=q5_reframe[:250],
            breakthrough_notes=f"Completed Practice Studio exercise: {payload.exercise_title[:100]}",
        )
        db.add(episode)

        profile = clinical_memory_service.get_or_create_profile(payload.user_id, db)
        reframes = list(profile.effective_reframes or [])
        if q5_reframe not in reframes:
            reframes.append(q5_reframe[:200])
            profile.effective_reframes = reframes[-10:]

        dist_map = dict(profile.dominant_distortions or {})
        dist_map[payload.distortion_category] = dist_map.get(payload.distortion_category, 0) + 1
        profile.dominant_distortions = dist_map
        profile.updated_at = now
    except Exception as e:
        print(f"[WARNING] Clinical memory synchronization notice: {e}")

    db.commit()
    db.refresh(record)
    return record


@router.get("", response_model=List[CBTExerciseResponse])
def get_user_cbt_exercises(
    user_id: uuid.UUID = Query(..., description="ID of the user"),
    distortion_category: Optional[str] = Query(default=None, description="Optional filter by distortion"),
    db: Session = Depends(get_db)
):
    """Retrieves all completed practice exercises for a user."""
    query = db.query(CBTExerciseSubmission).filter(
        CBTExerciseSubmission.user_id == user_id,
        CBTExerciseSubmission.is_completed == True
    )
    if distortion_category:
        query = query.filter(CBTExerciseSubmission.distortion_category == distortion_category)

    records = query.order_by(CBTExerciseSubmission.completed_at.desc()).all()
    return records


@router.get("/progress", response_model=CBTExerciseProgressSummary)
def get_cbt_exercises_progress(
    user_id: uuid.UUID = Query(..., description="ID of the user"),
    db: Session = Depends(get_db)
):
    """
    Computes overall and per-category progress across all 24 practice exercises.
    All numbers come directly from real database records.
    """
    submissions = (
        db.query(CBTExerciseSubmission)
        .filter(
            CBTExerciseSubmission.user_id == user_id,
            CBTExerciseSubmission.is_completed == True
        )
        .order_by(CBTExerciseSubmission.completed_at.desc())
        .all()
    )

    completed_exercise_ids = [s.exercise_id for s in submissions]
    total_completed = len(completed_exercise_ids)

    # Initialize categories progress dictionary with zero defaults
    categories_progress = {}
    for cat in ALL_DISTORTION_CATEGORIES:
        cat_count = sum(1 for s in submissions if s.distortion_category.lower() == cat.lower())
        cat_percentage = round((cat_count / 4.0) * 100.0, 1)
        categories_progress[cat] = CBTExerciseCategoryProgress(
            completed=cat_count,
            total=4,
            percentage=min(100.0, cat_percentage)
        )

    overall_percentage = round((total_completed / 24.0) * 100.0, 1) if total_completed > 0 else 0.0

    return CBTExerciseProgressSummary(
        total_completed=total_completed,
        total_available=24,
        completion_percentage=min(100.0, overall_percentage),
        categories_progress=categories_progress,
        completed_exercise_ids=completed_exercise_ids,
        recent_activity=submissions[:10]
    )


@router.get("/{exercise_id}", response_model=CBTExerciseResponse)
def get_exercise_detail(
    exercise_id: str,
    user_id: uuid.UUID = Query(..., description="ID of the user"),
    db: Session = Depends(get_db)
):
    """Retrieves previous answers for a specific exercise so user can review their work."""
    record = (
        db.query(CBTExerciseSubmission)
        .filter(
            CBTExerciseSubmission.user_id == user_id,
            CBTExerciseSubmission.exercise_id == exercise_id
        )
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail="Exercise submission not found.")
    return record

