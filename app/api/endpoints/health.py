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
    from app.services.llm.huggingface_provider import HuggingFaceLLMProvider
    import torch

    provider = llm_service.provider
    is_hf = isinstance(provider, HuggingFaceLLMProvider)

    return {
        "provider_type": type(provider).__name__,
        "is_available": provider.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "local_pipeline_active": getattr(provider, "local_pipeline", None) is not None,
        "model_repo": getattr(provider, "repo_id", None),
        "init_error": getattr(provider, "init_error", None),
    }

