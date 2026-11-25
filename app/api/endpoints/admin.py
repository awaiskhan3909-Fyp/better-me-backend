from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User
from app.api.deps import require_admin
from app.services.admin_analytics_service import AdminAnalyticsService

router = APIRouter(prefix="/admin", tags=["Admin Analytics & System Operations"])


@router.get("/overview")
def get_admin_overview(
    days: int = Query(default=30, description="Reporting window in days (7, 30, or 90)"),
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Returns high-level system KPIs, user activity trends, and practice metrics."""
    service = AdminAnalyticsService(db)
    return service.get_overview(days=days)


@router.get("/users")
def get_admin_users(
    days: int = Query(default=30, description="Reporting window in days (7, 30, or 90)"),
    limit: int = Query(default=50, ge=1, le=200, description="Max user records to return"),
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Returns pseudonymized user cohort analytics.
    Privacy guarantee: Never returns raw emails, real names, or message contents.
    """
    service = AdminAnalyticsService(db)
    return service.get_users_cohort(days=days, limit=limit)


@router.get("/safety")
def get_admin_safety(
    days: int = Query(default=30, description="Reporting window in days (7, 30, or 90)"),
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Returns safety classifier risk level distribution, alert trends, and crisis intervention statistics."""
    service = AdminAnalyticsService(db)
    return service.get_safety_report(days=days)


@router.get("/distortions")
def get_admin_distortions(
    days: int = Query(default=30, description="Reporting window in days (7, 30, or 90)"),
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Returns cognitive distortion prevalence, classifier confidence distribution, and Socratic gating metrics."""
    service = AdminAnalyticsService(db)
    return service.get_distortions_report(days=days)


@router.get("/sessions-funnel")
def get_admin_sessions_funnel(
    days: int = Query(default=30, description="Reporting window in days (7, 30, or 90)"),
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Returns therapy stage progression, turn counts per stage, and drop-off funnel across CBT session stages."""
    service = AdminAnalyticsService(db)
    return service.get_sessions_funnel(days=days)


@router.get("/ai-ops")
def get_admin_ai_ops(
    days: int = Query(default=30, description="Reporting window in days (7, 30, or 90)"),
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Returns LLM vs CBT template response source distributions, model usage, and offline fallback rates."""
    service = AdminAnalyticsService(db)
    return service.get_ai_ops(days=days)


@router.get("/engagement")
def get_admin_engagement(
    days: int = Query(default=30, description="Reporting window in days (7, 30, or 90)"),
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Returns user retention, intake distress baseline distributions, belief restructuring shifts, and relief usage."""
    service = AdminAnalyticsService(db)
    return service.get_engagement(days=days)


@router.get("/system")
def get_admin_system(
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Returns total row counts across all 10 system tables, database health check, and ML model load statuses."""
    service = AdminAnalyticsService(db)
    return service.get_system_status()
