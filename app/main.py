import app.db.database
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router
from app.services.distortion_service import distortion_service
from app.services.safety_service import safety_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML models into memory only once
    print("\n" + "=" * 50)
    print("[INFO] Starting Better Me AI Backend Services...")
    print("=" * 50)
    distortion_service.load_model()
    safety_service.load_model()
    print("=" * 50)
    print("[SUCCESS] All AI models loaded & ready for requests!")
    print("=" * 50 + "\n")
    yield
    print("[INFO] Shutting down Better Me AI Backend...")


app = FastAPI(
    title="Better Me AI Backend",
    description="AI-based Cognitive Distortion Detection and CBT Support System API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware setup for React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/")
def root():
    return {
        "name": "Better Me AI Backend",
        "status": "online",
        "docs_url": "/docs",
        "health_check": "/api/health"
    }
