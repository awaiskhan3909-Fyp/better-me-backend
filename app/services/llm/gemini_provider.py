import time
import requests
import logging
from typing import Optional

from app.core.config import GEMINI_API_KEY, LLM_MODEL_NAME
from app.services.llm.provider_interface import BaseLLMProvider, ProviderLLMResponse

logger = logging.getLogger("better_me.gemini_provider")


class GeminiLLMProvider(BaseLLMProvider):
    """
    Google Gemini API Provider implementation via direct Google REST API.
    Supports gemini-flash-latest, gemini-2.5-flash, etc. without external typing conflicts.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        configured_model = model_name or LLM_MODEL_NAME or "gemini-3.1-flash-lite"
        # Normalize model string for endpoint URI
        if not configured_model.startswith("models/"):
            self.model_uri = f"models/{configured_model}"
            self.model_name = configured_model
        else:
            self.model_uri = configured_model
            self.model_name = configured_model.replace("models/", "")
            
        logger.info(f"Initialized Gemini Provider with model '{self.model_name}' (URI: {self.model_uri})")

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def generate(self, prompt: str, system_instruction: str, temperature: float = 0.7, max_tokens: int = 512) -> ProviderLLMResponse:
        if not self.is_available():
            raise RuntimeError("Gemini LLM Provider is not available or API key missing.")

        start_time = time.time()
        
        # Primary endpoint attempts (fallback chain across fast, quota-available models)
        candidate_models = [
            self.model_uri,
            "models/gemini-3.1-flash-lite",
            "models/gemini-3.5-flash-lite",
            "models/gemini-3.8-flash",
            "models/gemini-flash-latest"
        ]
        # Remove duplicates while preserving order
        candidate_models = list(dict.fromkeys(candidate_models))

        last_exception = None
        response_json = None
        used_model = self.model_name

        for target_model in candidate_models:
            endpoint_url = f"https://generativelanguage.googleapis.com/v1beta/{target_model}:generateContent?key={self.api_key}"
            
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }
                ],
                "systemInstruction": {
                    "parts": [{"text": system_instruction}]
                },
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens
                }
            }

            try:
                resp = requests.post(endpoint_url, json=payload, timeout=12)
                if resp.status_code == 200:
                    response_json = resp.json()
                    used_model = target_model.replace("models/", "")
                    break
                else:
                    logger.warning(f"Model URI '{target_model}' returned status {resp.status_code}: {resp.text[:200]}")
                    last_exception = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                logger.warning(f"Failed request to '{target_model}': {e}")
                last_exception = e

        if not response_json:
            raise RuntimeError(f"All Gemini model endpoint attempts failed: {last_exception}")

        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        try:
            candidates = response_json.get("candidates", [])
            if not candidates:
                raise ValueError("Gemini API returned response without candidates.")

            content_parts = candidates[0].get("content", {}).get("parts", [])
            if not content_parts:
                raise ValueError("Gemini API candidate content parts empty.")

            raw_text = content_parts[0].get("text", "").strip()
            if not raw_text:
                raise ValueError("Gemini API generated empty text.")

            usage = response_json.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount")
            completion_tokens = usage.get("candidatesTokenCount")

            return ProviderLLMResponse(
                raw_content=raw_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                model_name=used_model,
                provider_name="google_gemini"
            )

        except Exception as parse_err:
            raise ValueError(f"Failed to parse Gemini API response: {parse_err}")
