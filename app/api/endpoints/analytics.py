import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories.user_repository import UserRepository
from app.db.repositories.analytics_repository import AnalyticsRepository
from app.schemas.auth import DashboardStatsResponse

router = APIRouter(prefix="/dashboard", tags=["Dashboard & Analytics"])


@router.get("/stats", response_model=DashboardStatsResponse)
def get_user_dashboard_stats(
    user_id: Optional[uuid.UUID] = Query(default=None, description="User UUID to fetch stats for. Defaults to demo user if omitted."),
    db: Session = Depends(get_db)
):
    user_repo = UserRepository(db)
    if user_id:
        user = user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")
        target_user_id = user.id
    else:
        demo_user = user_repo.get_or_create_default_user()
        target_user_id = demo_user.id

    analytics_repo = AnalyticsRepository(db)
    stats = analytics_repo.get_dashboard_stats(target_user_id)
    return stats
