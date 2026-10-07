from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Better Me AI Backend",
        "version": "1.0.0"
    }
