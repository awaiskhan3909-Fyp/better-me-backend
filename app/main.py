import logging
import app.db.database
from app.db.database import engine, Base
import app.db.models
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router
from app.core.config import ALLOWED_ORIGINS, ALLOW_ORIGIN_REGEX
from app.services.distortion_service import distortion_service
from app.services.safety_service import safety_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Auto-create DB tables if not present
    Base.metadata.create_all(bind=engine)
    # Load ML models into memory only once
    logger.info("Starting Better Me AI Backend Services...")
    distortion_service.load_model()
    safety_service.load_model()
    logger.info("All AI models loaded & ready for requests!")
    yield
    logger.info("Shutting down Better Me AI Backend...")


app = FastAPI(
    title="Better Me AI Backend",
    description="AI-based Cognitive Distortion Detection and CBT Support System API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware setup for React frontend integration (Local, Vercel & Tunnel URLs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOW_ORIGIN_REGEX if ALLOW_ORIGIN_REGEX else None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Better Me AI Backend",
        "version": "1.0.0"
    }


@app.get("/")
def root():
    return {
        "name": "Better Me AI Backend",
        "status": "online",
        "docs_url": "/docs",
        "health_check": "/api/health"
    }
