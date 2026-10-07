from fastapi import APIRouter
from app.api.endpoints import health, analyze, conversations

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(analyze.router, tags=["Analysis"])
api_router.include_router(conversations.router, tags=["Conversations"])
