from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel


class ProviderLLMResponse(BaseModel):
    raw_content: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: float
    model_name: str
    provider_name: str


class BaseLLMProvider(ABC):
    """
    Abstract Base Class for Conversational LLM Providers.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if provider credentials and client are properly configured."""
        pass

    @abstractmethod
    def generate(self, prompt: str, system_instruction: str, temperature: float = 0.7, max_tokens: int = 512) -> ProviderLLMResponse:
        """Executes prompt generation against the LLM provider."""
        pass
