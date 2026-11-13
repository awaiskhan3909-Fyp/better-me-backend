import pytest
from unittest.mock import patch, MagicMock

from app.services.llm.gemini_provider import GeminiLLMProvider
from app.services.llm.provider_interface import ProviderLLMResponse


def test_generate_accepts_full_arguments_and_constructs_payload():
    provider = GeminiLLMProvider(api_key="fake_test_key", model_name="gemini-flash-lite-latest")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "I hear how anxious you are feeling about the exam. What feels most daunting?"}]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 42,
            "candidatesTokenCount": 18
        }
    }

    recent_history = "Patient: Hello, I have an exam tomorrow.\nBetter Me: It is understandable to be nervous."

    with patch("requests.post", return_value=mock_resp) as mock_post:
        resp = provider.generate(
            prompt="Now I feel completely frozen and unprepared.",
            system_instruction="You are a clinical CBT companion.",
            temperature=0.7,
            max_tokens=240,
            user_text="Now I feel completely frozen and unprepared.",
            strategy="empathetic_validation",
            distortion_name="Catastrophizing",
            is_roman_urdu=False,
            recent_history=recent_history,
            extra_param="unused"
        )

        assert mock_post.called
        call_args = mock_post.call_args
        endpoint_url = call_args[0][0]
        headers = call_args[1].get("headers", {})
        payload = call_args[1].get("json", {})

        # 1. API key in header, not in URL
        assert "key=" not in endpoint_url
        assert headers.get("x-goog-api-key") == "fake_test_key"
        assert endpoint_url.endswith("/models/gemini-flash-lite-latest:generateContent")

        # 2. Multi-turn contents structure
        contents = payload.get("contents", [])
        assert len(contents) == 3
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"][0]["text"] == "Hello, I have an exam tomorrow."
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"][0]["text"] == "It is understandable to be nervous."
        assert contents[2]["role"] == "user"
        assert contents[2]["parts"][0]["text"] == "Now I feel completely frozen and unprepared."

        # 3. Response validation
        assert isinstance(resp, ProviderLLMResponse)
        assert resp.model_name == "gemini-flash-lite-latest"
        assert resp.provider_name == "google_gemini"
        assert "I hear how anxious you are feeling" in resp.raw_content
        assert resp.prompt_tokens == 42
        assert resp.completion_tokens == 18


def test_generate_retries_on_503_and_succeeds():
    provider = GeminiLLMProvider(api_key="fake_test_key", model_name="gemini-flash-lite-latest")

    mock_503 = MagicMock()
    mock_503.status_code = 503
    mock_503.text = "Service Temporarily Unavailable"

    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Recovered after retry."}]
                }
            }
        ]
    }

    with patch("requests.post", side_effect=[mock_503, mock_200]) as mock_post:
        with patch("time.sleep"):  # Skip sleep during tests
            resp = provider.generate(
                prompt="Testing retry",
                system_instruction="Be concise"
            )

            assert mock_post.call_count == 2
            assert resp.raw_content == "Recovered after retry."
