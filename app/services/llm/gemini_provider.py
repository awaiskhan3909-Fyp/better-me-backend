import time
import requests
import logging
from typing import Optional, List, Dict, Any

from app.core.config import GEMINI_API_KEY, GEMINI_MODEL
from app.services.llm.provider_interface import BaseLLMProvider, ProviderLLMResponse

logger = logging.getLogger("better_me.gemini_provider")


class GeminiLLMProvider(BaseLLMProvider):
    """
    Google Gemini API Provider implementation via direct Google REST API.
    Sends API key in x-goog-api-key header and uses only the configured GEMINI_MODEL.
    Supports multi-turn contents format, 12s timeout, and single retry on 429/5xx.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = (api_key or GEMINI_API_KEY or "").strip()
        configured_model = (model_name or GEMINI_MODEL or "").strip()
        if configured_model:
            if configured_model.startswith("models/"):
                self.model_name = configured_model.replace("models/", "")
            else:
                self.model_name = configured_model
        else:
            self.model_name = ""

        logger.info(f"Initialized Gemini Provider with model '{self.model_name}'")

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def _build_contents(self, prompt: str, recent_history: Optional[str] = None) -> List[Dict[str, Any]]:
        raw_turns = []
        if recent_history and recent_history.strip():
            lines = recent_history.strip().split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("Patient:") or line.startswith("User:"):
                    prefix_len = len("Patient:") if line.startswith("Patient:") else len("User:")
                    msg_text = line[prefix_len:].strip()
                    if msg_text:
                        raw_turns.append({"role": "user", "parts": [{"text": msg_text}]})
                elif line.startswith("Better Me:") or line.startswith("Assistant:") or line.startswith("AI:"):
                    if line.startswith("Better Me:"):
                        prefix_len = len("Better Me:")
                    elif line.startswith("Assistant:"):
                        prefix_len = len("Assistant:")
                    else:
                        prefix_len = len("AI:")
                    msg_text = line[prefix_len:].strip()
                    if msg_text:
                        raw_turns.append({"role": "model", "parts": [{"text": msg_text}]})

        # Merge consecutive turns of identical role
        contents = []
        for turn in raw_turns:
            if contents and contents[-1]["role"] == turn["role"]:
                contents[-1]["parts"][0]["text"] += "\n" + turn["parts"][0]["text"]
            else:
                contents.append(turn)

        # Append current user prompt as final turn
        if contents and contents[-1]["role"] == "user":
            contents[-1]["parts"][0]["text"] += "\n\n" + prompt
        else:
            contents.append({"role": "user", "parts": [{"text": prompt}]})

        return contents

    FALLBACK_MODELS = [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.1-flash-lite",
        "gemini-3-flash-preview",
    ]

    def generate(
        self,
        prompt: str,
        system_instruction: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        user_text: Optional[str] = None,
        strategy: Optional[str] = None,
        distortion_name: Optional[str] = None,
        is_roman_urdu: bool = False,
        recent_history: Optional[str] = None,
        **kwargs
    ) -> ProviderLLMResponse:
        if not self.is_available():
            raise RuntimeError("Gemini LLM Provider is not available or API key missing.")

        primary_slug = (self.model_name or GEMINI_MODEL or "gemini-3.5-flash").strip()
        if primary_slug.startswith("models/"):
            primary_slug = primary_slug.replace("models/", "")

        # Build deduplicated ordered failover chain so 429/503 on one model fails over immediately
        candidate_models: List[str] = []
        for m in [primary_slug, *self.FALLBACK_MODELS]:
            clean_m = m.replace("models/", "").strip()
            if clean_m and clean_m not in candidate_models:
                candidate_models.append(clean_m)

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        contents = self._build_contents(prompt=prompt, recent_history=recent_history)

        start_time = time.time()
        response_json = None
        used_model = primary_slug
        last_exception = None

        for model_slug in candidate_models:
            endpoint_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_slug}:generateContent"
            generation_config: Dict[str, Any] = {
                "temperature": temperature,
                "maxOutputTokens": max(max_tokens, 2048)
            }

            payload = {
                "contents": contents,
                "systemInstruction": {
                    "parts": [{"text": system_instruction}]
                },
                "generationConfig": generation_config
            }

            try:
                resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=12)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
                    # Extract non-thought text part cleanly (handles both standard and thinking models)
                    text_parts = [p.get("text", "") for p in parts if p.get("text") and not p.get("thought")]
                    raw_text_check = "\n".join(text_parts).strip() if text_parts else (parts[0].get("text", "").strip() if parts else "")
                    if raw_text_check:
                        response_json = data
                        used_model = model_slug
                        break
                    last_exception = ValueError(f"Empty candidate text from '{model_slug}'")
                    logger.warning(f"Gemini model '{model_slug}' returned empty text; failing over to next model.")
                    continue

                if resp.status_code in (400, 429, 404, 500, 502, 503, 504):
                    logger.warning(
                        f"Gemini API returned status {resp.status_code} for '{model_slug}': {resp.text[:160]}. Failing over to next model..."
                    )
                    last_exception = RuntimeError(f"HTTP {resp.status_code} ({model_slug}): {resp.text[:200]}")
                    continue

                logger.warning(f"Gemini API returned non-retryable status {resp.status_code}: {resp.text[:200]}")
                last_exception = RuntimeError(f"HTTP {resp.status_code} ({model_slug}): {resp.text[:200]}")
                break

            except requests.RequestException as e:
                logger.warning(f"Gemini API request exception for '{model_slug}' ({e}). Failing over to next model...")
                last_exception = e
                continue

        if not response_json:
            raise RuntimeError(f"Gemini API call failed across models {candidate_models}: {last_exception}")

        latency_ms = round((time.time() - start_time) * 1000, 2)

        try:
            candidates = response_json.get("candidates", [])
            if not candidates:
                raise ValueError(f"Gemini API response contained no candidates: {response_json}")

            content_parts = candidates[0].get("content", {}).get("parts", [])
            if not content_parts:
                raise ValueError("Gemini API candidate content parts empty.")

            text_parts = [p.get("text", "") for p in content_parts if p.get("text") and not p.get("thought")]
            raw_text = "\n".join(text_parts).strip() if text_parts else content_parts[0].get("text", "").strip()
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
