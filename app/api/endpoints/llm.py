from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.services.llm_service import llm_service

router = APIRouter(prefix="/llm", tags=["LLM Inference Engine"])


class GenerateLLMRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 150


class GenerateLLMResponse(BaseModel):
    generated_text: str
    model_name: str
    provider_name: str
    latency_ms: float


@router.post("/generate", response_model=GenerateLLMResponse)
async def generate_from_llm(req: GenerateLLMRequest):
    """
    Direct LLM generation endpoint.
    Used when running on GPU (Kaggle/Colab) to serve fine-tuned 4-bit LLaMA-3 CBT inferences.
    """
    provider = llm_service.provider
    resp = provider.generate(
        prompt=req.prompt,
        temperature=req.temperature or 0.7,
        max_tokens=req.max_tokens or 150
    )
    return GenerateLLMResponse(
        generated_text=resp.raw_content,
        model_name=resp.model_name,
        provider_name=resp.provider_name,
        latency_ms=resp.latency_ms
    )

