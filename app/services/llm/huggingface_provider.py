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
        if getattr(self, "unsloth_model", None) is not None:
            logger.info("Unsloth model already initialized in memory.")
            return

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
        if getattr(self, "unsloth_model", None) is not None or self.local_pipeline is not None:
            return True
        return bool(self.hf_token and self.hf_token.strip())

    def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_tokens: int = 180,
        user_text: Optional[str] = None,
        distortion_name: Optional[str] = None,
        is_roman_urdu: bool = False,
        recent_history: Optional[str] = None,
        strategy: Optional[str] = None,
        **kwargs
    ) -> ProviderLLMResponse:
        start_time = time.time()

        # Check if input is a simple greeting or casual conversational opener
        clean_user_input = user_text.lower().strip() if user_text else ""
        is_greeting = (
            strategy == "normal_conversation"
            or clean_user_input in ["hi", "hello", "hey", "salam", "assalam", "assalamoalaikum", "assalamualaykum", "good morning", "good afternoon", "good evening", "howdy", "sup", "yo"]
            or any(clean_user_input.startswith(g) for g in ["hi ", "hello ", "hey ", "salam ", "assalam "])
        )

        target_distortion = distortion_name if distortion_name and distortion_name not in ["None", "Normal", "no distortion"] else None

        if is_greeting:
            if is_roman_urdu:
                instruction = (
                    "You are Better Me, a warm and welcoming clinical CBT companion. "
                    "Respond with a warm, caring greeting in natural Roman Urdu (1-2 sentences). "
                    "Welcome the user gently and ask how they are feeling today."
                )
            else:
                instruction = (
                    "You are Better Me, a warm and welcoming clinical CBT companion. "
                    "Respond to the patient's greeting with kindness and warmth in 1-2 friendly sentences. "
                    "Welcome them and gently ask how they are doing or what brings them here today."
                )
            patient_input = user_text if user_text else "Hello"
            alpaca_prompt = (
                f"Below is an instruction that describes a welcoming clinical companion greeting task, paired with a greeting from the patient. "
                f"Write a warm, gentle therapist greeting (1-2 sentences maximum).\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{patient_input}\n\n"
                f"### Response:\n"
            )
        else:
            if is_roman_urdu:
                if target_distortion:
                    instruction = (
                        f"You are Better Me, a clinical CBT companion. The patient exhibits {target_distortion} and is speaking in Roman Urdu. "
                        f"Respond warmly and empathetically in natural, conversational Roman Urdu (Latin script Urdu). "
                        f"Keep your response concise (under 3-4 sentences). Do NOT interrogate the patient with multiple questions. "
                        f"Validate their distress, and conclude with exactly ONE gentle Socratic question to help them examine this thought."
                    )
                else:
                    instruction = (
                        "You are Better Me, an empathetic clinical CBT companion. The patient is speaking in Roman Urdu. "
                        "Respond with warmth, active listening, and calm emotional validation in conversational Roman Urdu (2-3 sentences). "
                        "Do NOT overwhelm with multiple questions. Ask only ONE supportive question at the end."
                    )
            else:
                if target_distortion:
                    instruction = (
                        f"You are Better Me, a clinical CBT companion. The patient exhibits {target_distortion}. "
                        f"Validate their emotional distress with deep empathy in 2-3 sentences. "
                        f"Do NOT bombard or interrogate the patient with multiple questions. "
                        f"Conclude with exactly ONE focused, open-ended Socratic question to help them reflect on this thought."
                    )
                else:
                    instruction = (
                        "You are Better Me, a clinical CBT companion. "
                        "Provide compassionate validation and thoughtful active listening in 2-3 sentences. "
                        "Never ask multiple questions in one reply. Ask only ONE gentle, open-ended question at the end to explore how they are feeling."
                    )

            # Input is focused on the patient's distressed thought
            if user_text:
                if recent_history:
                    patient_input = f"{recent_history}\nCurrent Thought: {user_text}"
                else:
                    patient_input = user_text
            else:
                patient_input = prompt

            alpaca_prompt = (
                f"Below is an instruction that describes a clinical CBT therapeutic task, paired with an input containing the patient's distressed thought. "
                f"Write a direct, compassionate therapist response that provides empathetic validation and ends with a single Socratic question. "
                f"Keep it concise, supportive, and natural without repetitive questions, notes, or roleplay commentary.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{patient_input}\n\n"
                f"### Response:\n"
            )

        # 1. Unsloth Native GPU Execution (Primary)
        if getattr(self, "unsloth_model", None) is not None and getattr(self, "unsloth_tokenizer", None) is not None:
            try:
                inputs = self.unsloth_tokenizer([alpaca_prompt], return_tensors="pt").to("cuda")
                input_len = inputs["input_ids"].shape[1]

                pad_id = self.unsloth_tokenizer.pad_token_id
                if pad_id is None:
                    eos_id = self.unsloth_tokenizer.eos_token_id
                    pad_id = eos_id[0] if isinstance(eos_id, (list, tuple)) else eos_id

                gen_kwargs = {
                    "max_new_tokens": min(max_tokens, 150),
                    "use_cache": True,
                    "repetition_penalty": 1.18,
                }
                if isinstance(pad_id, int):
                    gen_kwargs["pad_token_id"] = pad_id
                
                # Multi-EOS token registration (handles standard EOS 128001 + LLaMA-3 EOT 128009)
                eos_candidates = []
                if self.unsloth_tokenizer.eos_token_id is not None:
                    if isinstance(self.unsloth_tokenizer.eos_token_id, int):
                        eos_candidates.append(self.unsloth_tokenizer.eos_token_id)
                    elif isinstance(self.unsloth_tokenizer.eos_token_id, (list, tuple)):
                        eos_candidates.extend(self.unsloth_tokenizer.eos_token_id)
                for special_eos in [128001, 128009]:
                    if special_eos not in eos_candidates:
                        eos_candidates.append(special_eos)
                if eos_candidates:
                    gen_kwargs["eos_token_id"] = eos_candidates

                if temperature and temperature > 0:
                    gen_kwargs["temperature"] = float(temperature)
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["top_p"] = 0.9

                outputs = self.unsloth_model.generate(
                    **inputs,
                    **gen_kwargs
                )

                # Slice newly generated tokens directly
                generated_tokens = outputs[0][input_len:]
                generated_text = self.unsloth_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                if not generated_text:
                    raw_decoded = self.unsloth_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                    if "### Response:" in raw_decoded:
                        generated_text = raw_decoded.split("### Response:")[-1].strip()
                    else:
                        generated_text = raw_decoded.strip()

                latency_ms = (time.time() - start_time) * 1000
                print(f"✅ [LLM GPU] Live LLaMA-3 generation complete in {latency_ms:.0f}ms! (Generated tokens: {len(generated_tokens)})")
                return ProviderLLMResponse(
                    raw_content=generated_text,
                    prompt_tokens=input_len,
                    completion_tokens=len(generated_tokens),
                    latency_ms=latency_ms,
                    model_name=self.repo_id,
                    provider_name="unsloth_llama3_lora_gpu"
                )
            except Exception as unsloth_gen_err:
                import traceback
                print(f"❌ [LLM GPU Generation Error]: {unsloth_gen_err}")
                traceback.print_exc()
                logger.error(f"Unsloth generation error: {unsloth_gen_err}")
                raise unsloth_gen_err

        # 2. Local Pipeline Execution (Standard Transformers)
        if self.local_pipeline is not None and self.local_pipeline != "unsloth":
            try:
                outputs = self.local_pipeline(
                    alpaca_prompt,
                    max_new_tokens=min(max_tokens, 150),
                    temperature=temperature,
                    repetition_penalty=1.18,
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
                raise e

        # 3. Remote Hugging Face Inference API Execution (Only if local not loaded)
        if self.hf_token and not self.repo_id.startswith("/"):
            api_url = f"https://api-inference.huggingface.co/models/{self.repo_id}"
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": alpaca_prompt,
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

        raise RuntimeError("No available LLM execution pipeline found.")
