import time
import os
import logging
import requests
from typing import Optional

from app.core.config import HF_DISTORTION_REPO
from app.services.llm.provider_interface import BaseLLMProvider, ProviderLLMResponse

logger = logging.getLogger("better_me.huggingface_provider")

DEFAULT_CBT_HF_REPO = os.getenv("HF_LLM_REPO", "awaiskhan4039/better-me-cbt-llama3-lora")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_TOKEN", None))


class HuggingFaceLLMProvider(BaseLLMProvider):
    """
    Open-Source LLM Provider supporting fine-tuned Llama-3 / Qwen models.
    Supports two operating modes:
    1. Remote Serverless Hugging Face Inference API (Zero GPU VRAM needed).
    2. Local In-Memory 4-bit Quantized Pipeline (Direct on Colab T4 GPU).
    """

    def __init__(
        self,
        repo_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        use_local_pipeline: bool = False
    ):
        self.repo_id = repo_id or DEFAULT_CBT_HF_REPO
        self.hf_token = hf_token or HF_TOKEN
        self.use_local_pipeline = use_local_pipeline
        self.local_pipeline = None

        # Auto-detect CUDA GPU environment (e.g. Google Colab Tesla T4)
        if not self.use_local_pipeline:
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("CUDA GPU detected. Automatically enabling local 4-bit HuggingFace pipeline on GPU.")
                    self.use_local_pipeline = True
            except Exception:
                pass

        if self.use_local_pipeline:
            self._init_local_pipeline()

        logger.info(f"Initialized HuggingFace LLM Provider with repo '{self.repo_id}' (Local: {self.use_local_pipeline})")


    def _init_local_pipeline(self):
        """Loads 4-bit model directly on available GPU."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

            logger.info(f"Loading local 4-bit pipeline for '{self.repo_id}' on CUDA...")
            tokenizer = AutoTokenizer.from_pretrained(self.repo_id, token=self.hf_token)

            model = None
            try:
                from peft import AutoPeftModelForCausalLM
                model = AutoPeftModelForCausalLM.from_pretrained(
                    self.repo_id,
                    token=self.hf_token,
                    torch_dtype=torch.float16,
                    load_in_4bit=True,
                    device_map="auto"
                )
            except Exception as peft_err:
                logger.info(f"Loading as standard CausalLM: {peft_err}")
                model = AutoModelForCausalLM.from_pretrained(
                    self.repo_id,
                    token=self.hf_token,
                    torch_dtype=torch.float16,
                    load_in_4bit=True,
                    device_map="auto"
                )

            self.local_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
            )
            logger.info("Local HuggingFace 4-bit pipeline ready!")
        except Exception as e:
            logger.warning(f"Could not initialize local pipeline: {e}. Falling back to remote HF API.")
            self.local_pipeline = None

    def is_available(self) -> bool:
        if self.local_pipeline is not None:
            return True
        return bool(self.hf_token and self.hf_token.strip())

    def generate(
        self,
        prompt: str,
        system_instruction: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> ProviderLLMResponse:
        start_time = time.time()

        # Format full prompt with system instruction
        full_prompt = f"System: {system_instruction}\n\n{prompt}"

        # 1. Local Pipeline Execution
        if self.local_pipeline is not None:
            try:
                outputs = self.local_pipeline(
                    full_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    return_full_text=False
                )
                generated_text = outputs[0]["generated_text"].strip()
                latency_ms = (time.time() - start_time) * 1000

                return ProviderLLMResponse(
                    raw_content=generated_text,
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(generated_text.split()),
                    latency_ms=latency_ms,
                    model_name=self.repo_id,
                    provider_name="local_huggingface_gpu"
                )
            except Exception as e:
                logger.error(f"Local pipeline generation error: {e}")

        # 2. Remote Hugging Face Inference API Execution
        api_url = f"https://api-inference.huggingface.co/models/{self.repo_id}"
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Hugging Face API returned status {response.status_code}: {response.text}")

        res_data = response.json()
        if isinstance(res_data, list) and len(res_data) > 0:
            generated_text = res_data[0].get("generated_text", "")
        elif isinstance(res_data, dict):
            generated_text = res_data.get("generated_text", "")
        else:
            generated_text = str(res_data)

        latency_ms = (time.time() - start_time) * 1000
        return ProviderLLMResponse(
            raw_content=generated_text.strip(),
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(generated_text.split()),
            latency_ms=latency_ms,
            model_name=self.repo_id,
            provider_name="huggingface_inference_api"
        )
