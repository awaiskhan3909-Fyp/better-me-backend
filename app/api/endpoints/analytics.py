import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories.user_repository import UserRepository
from app.db.repositories.analytics_repository import AnalyticsRepository
from app.schemas.auth import DashboardStatsResponse

router = APIRouter(tags=["Dashboard & Analytics"])


@router.get("/dashboard/stats", response_model=DashboardStatsResponse)
@router.get("/stats", response_model=DashboardStatsResponse)
def get_user_dashboard_stats(
    user_id: Optional[str] = Query(default=None, description="User UUID to fetch stats for."),
    db: Session = Depends(get_db)
):
    user_repo = UserRepository(db)
    analytics_repo = AnalyticsRepository(db)
    if user_id:
        try:
            parsed_uuid = uuid.UUID(str(user_id))
            user = user_repo.get_by_id(parsed_uuid)
            if user:
                return analytics_repo.get_dashboard_stats(user.id)
        except (ValueError, TypeError):
            pass
        return DashboardStatsResponse(
            user_name="User",
            total_sessions=0,
            avg_duration_minutes=0,
            total_messages=0,
            thoughts_analyzed=0,
            categories_practiced=0,
            current_risk_level="Safe",
            primary_focus=[],
            primary_goal=None,
            distortions_breakdown={},
            recent_sessions=[],
            emotional_trends=[]
        )

    demo_user = user_repo.get_or_create_default_user()
    return analytics_repo.get_dashboard_stats(demo_user.id)
