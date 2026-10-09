from fastapi import APIRouter
from app.api.endpoints import health, analyze, conversations, auth, intake, analytics, thought_records

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(auth.router, tags=["Authentication"])
api_router.include_router(intake.router, tags=["Intake Assessment"])
api_router.include_router(analytics.router, tags=["Dashboard & Analytics"])
api_router.include_router(analyze.router, tags=["Analysis"])
api_router.include_router(conversations.router, tags=["Conversations"])
api_router.include_router(thought_records.router, tags=["CBT Thought Records"])
