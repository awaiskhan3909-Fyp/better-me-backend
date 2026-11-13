from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Better Me AI Backend",
        "version": "1.0.0"
    }


@router.get("/llm-status")
def llm_status():
    from app.services.llm_service import llm_service
    from app.core.config import GEMINI_API_KEY

    provider = llm_service.provider
    key = getattr(provider, "api_key", None) or GEMINI_API_KEY

    return {
        "provider": type(provider).__name__,
        "model": getattr(provider, "model_name", None),
        "key_configured": bool(key and str(key).strip()),
        "cuda_available": False,
        "gpu_name": None,
    }

