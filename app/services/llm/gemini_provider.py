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

        model_slug = (self.model_name or GEMINI_MODEL or "").strip()
        if not model_slug:
            raise RuntimeError("GEMINI_MODEL is not set in environment or configuration.")
        if model_slug.startswith("models/"):
            model_slug = model_slug.replace("models/", "")

        endpoint_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_slug}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        contents = self._build_contents(prompt=prompt, recent_history=recent_history)

        generation_config: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max(max_tokens, 1024)
        }
        if "gemini-3" in model_slug or "gemini-2.5" in model_slug:
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}

        payload = {
            "contents": contents,
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            },
            "generationConfig": generation_config
        }

        start_time = time.time()
        response_json = None
        last_exception = None

        # Execute request with 12s timeout and single retry on 429/5xx
        for attempt in range(2):
            try:
                resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=12)
                if resp.status_code == 200:
                    response_json = resp.json()
                    break

                if resp.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                    logger.warning(
                        f"Gemini API returned retryable status {resp.status_code} for '{model_slug}'. Retrying once in 1s..."
                    )
                    time.sleep(1.0)
                    continue

                logger.warning(f"Gemini API returned status {resp.status_code}: {resp.text[:200]}")
                last_exception = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                break

            except requests.RequestException as e:
                if attempt == 0:
                    logger.warning(f"Gemini API request exception ({e}). Retrying once in 1s...")
                    time.sleep(1.0)
                    continue
                last_exception = e
                break

        if not response_json:
            raise RuntimeError(f"Gemini API call failed for model '{model_slug}': {last_exception}")

        latency_ms = round((time.time() - start_time) * 1000, 2)

        try:
            candidates = response_json.get("candidates", [])
            if not candidates:
                raise ValueError(f"Gemini API response contained no candidates: {response_json}")

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
                model_name=model_slug,
                provider_name="google_gemini"
            )

        except Exception as parse_err:
            raise ValueError(f"Failed to parse Gemini API response: {parse_err}")
