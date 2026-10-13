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
        self.init_error = None

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
        """Loads 4-bit model directly on available GPU with Unsloth optimization."""
        # 1. Try Unsloth FastLanguageModel first (Ultra-fast, native format for this trained model)
        try:
            import torch
            from unsloth import FastLanguageModel
            print(f"\n[LLM] Loading fine-tuned 4-bit LLaMA-3 LoRA '{self.repo_id}' on CUDA via Unsloth...")
            logger.info(f"Unsloth detected on CUDA! Loading fine-tuned 4-bit LLaMA-3 LoRA '{self.repo_id}' directly...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.repo_id,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
                token=self.hf_token
            )
            FastLanguageModel.for_inference(model)
            self.unsloth_model = model
            self.unsloth_tokenizer = tokenizer
            self.local_pipeline = "unsloth"
            self.init_error = None
            print("✅ [LLM] Unsloth 4-bit LLaMA-3 LoRA model successfully loaded on GPU!")
            logger.info("✅ Unsloth 4-bit LLaMA-3 LoRA model ready on GPU!")
            return
        except Exception as unsloth_err:
            print(f"⚠️ [LLM] Unsloth init notice: {unsloth_err}. Trying standard transformers pipeline...")
            logger.info(f"Unsloth initialization not available ({unsloth_err}). Trying standard transformers pipeline...")

        # 2. Fallback to standard Hugging Face Transformers + PEFT
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

            logger.info(f"Loading local 4-bit pipeline for '{self.repo_id}' on CUDA via standard transformers...")
            tokenizer = AutoTokenizer.from_pretrained(self.repo_id, token=self.hf_token)

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )

            model = None
            try:
                from peft import AutoPeftModelForCausalLM
                model = AutoPeftModelForCausalLM.from_pretrained(
                    self.repo_id,
                    token=self.hf_token,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            except Exception as peft_err:
                logger.info(f"Loading as standard CausalLM: {peft_err}")
                model = AutoModelForCausalLM.from_pretrained(
                    self.repo_id,
                    token=self.hf_token,
                    quantization_config=quantization_config,
                    device_map="auto"
                )

            self.local_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
            )
            self.init_error = None
            logger.info("✅ Standard HuggingFace 4-bit pipeline ready!")
        except Exception as e:
            self.init_error = str(e)
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
        max_tokens: int = 150
    ) -> ProviderLLMResponse:
        start_time = time.time()

        # Format full prompt with the fine-tuned Alpaca CBT training format
        alpaca_prompt = (
            f"Below is an instruction that describes a clinical CBT therapeutic task, paired with an input containing the patient's distressed thought. Write a response that provides empathetic validation, Socratic cognitive restructuring, and a balanced reframe.\n\n"
            f"### Instruction:\n{system_instruction}\n\n"
            f"### Input:\n{prompt}\n\n"
            f"### Response:\n"
        )

        # 1. Unsloth Native GPU Execution
        if getattr(self, "unsloth_model", None) is not None and getattr(self, "unsloth_tokenizer", None) is not None:
            try:
                inputs = self.unsloth_tokenizer([alpaca_prompt], return_tensors="pt").to("cuda")
                outputs = self.unsloth_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.unsloth_tokenizer.eos_token_id
                )
                generated_full = self.unsloth_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                if "### Response:" in generated_full:
                    generated_text = generated_full.split("### Response:")[-1].strip()
                else:
                    generated_text = generated_full.strip()

                latency_ms = (time.time() - start_time) * 1000
                return ProviderLLMResponse(
                    raw_content=generated_text,
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(generated_text.split()),
                    latency_ms=latency_ms,
                    model_name=self.repo_id,
                    provider_name="unsloth_llama3_lora_gpu"
                )
            except Exception as unsloth_gen_err:
                logger.error(f"Unsloth generation error: {unsloth_gen_err}")

        # 2. Local Pipeline Execution (Standard Transformers)
        if self.local_pipeline is not None and self.local_pipeline != "unsloth":
            try:
                outputs = self.local_pipeline(
                    alpaca_prompt,
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

        response = requests.post(api_url, headers=headers, json=payload, timeout=15)
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
