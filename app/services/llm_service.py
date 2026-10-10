import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from app.core.config import LLM_FALLBACK_ENABLED, LLM_PROVIDER, HF_LLM_REPO, USE_LOCAL_LLM
from app.services.llm import (
    GeminiLLMProvider,
    HuggingFaceLLMProvider,
    prompt_builder,
    PROMPT_VERSION,
    response_parser,
    fallback_service
)
from app.services.conversation_manager import ConversationContext
from app.schemas.response import DistortionPrediction, EntityItem, CBTGuidance

logger = logging.getLogger("better_me.llm_service")


def get_default_provider():
    if LLM_PROVIDER in ["huggingface", "local_hf", "hf"]:
        logger.info(f"Using Open-Source HuggingFace Provider for CBT ({HF_LLM_REPO})")
        return HuggingFaceLLMProvider(
            repo_id=HF_LLM_REPO,
            use_local_pipeline=USE_LOCAL_LLM or (LLM_PROVIDER == "local_hf")
        )
    return GeminiLLMProvider()


class LLMResult(BaseModel):
    content: str
    response_source: str
    response_type: str
    model_name: str
    cbt_data: Optional[Dict[str, Any]] = None
    llm_metadata: Dict[str, Any]


class LLMService:
    """
    Unified Orchestrator for Conversational LLM Integration.
    Supports both Google Gemini API and Fine-Tuned Open-Source HuggingFace Models.
    Controls prompt construction, provider delegation, response validation,
    deterministic safety overrides, and seamless offline fallbacks.
    """

    def __init__(self, provider=None):
        self.provider = provider or get_default_provider()

    def generate_response(
        self,
        user_text: str,
        strategy: str,
        decision_reason: str,
        safety_risk_level: str,
        distortion_pred: Optional[DistortionPrediction] = None,
        entities: Optional[List[EntityItem]] = None,
        cbt_guidance: Optional[CBTGuidance] = None,
        conversation_context: Optional[ConversationContext] = None,
        clinical_memory: Optional[str] = None,
        is_roman_urdu: bool = False
    ) -> LLMResult:
        """
        Main entry point for generating strategy-guided conversational output.
        """
        # 1. STRICT SAFETY OVERRIDE GATE
        if strategy == "safety_response":
            fallback_text, _ = fallback_service.generate_fallback(
                user_text, strategy, cbt_guidance, is_roman_urdu=is_roman_urdu
            )
            return LLMResult(
                content=fallback_text,
                response_source="safety_override_engine",
                response_type="crisis_intervention",
                model_name="deterministic_safety_override",
                cbt_data=None,
                llm_metadata={
                    "prompt_version": PROMPT_VERSION,
                    "provider": "safety_override_engine",
                    "model": "deterministic_safety_override",
                    "strategy": strategy,
                    "fallback_used": False,
                    "reason": "deterministic_safety_override"
                }
            )

        # 2. Build Structured LLM Input Contract & Prompt Text
        contract = prompt_builder.build_contract(
            user_text=user_text,
            strategy=strategy,
            decision_reason=decision_reason,
            safety_risk_level=safety_risk_level,
            distortion_pred=distortion_pred,
            entities=entities,
            cbt_guidance=cbt_guidance,
            conversation_context=conversation_context,
            clinical_memory=clinical_memory,
            is_roman_urdu=is_roman_urdu
        )
        prompt_text = prompt_builder.build_prompt_text(contract)

        # 3. Check Provider Availability & Execute LLM Generation
        if self.provider.is_available():
            try:
                provider_resp = self.provider.generate(
                    prompt=prompt_text,
                    system_instruction=prompt_builder.system_instruction,
                    temperature=0.7,
                    max_tokens=512
                )

                # Validate LLM Output
                is_valid, sanitized_text, failure_reason = response_parser.validate_and_sanitize(
                    provider_resp.raw_content,
                    strategy
                )

                if is_valid:
                    # Determine CBT Data
                    cbt_data_json = None
                    if strategy == "cbt_support" and cbt_guidance:
                        cbt_data_json = cbt_guidance.model_dump()
                        cbt_data_json["balanced_thought_guidance"] = sanitized_text

                    return LLMResult(
                        content=sanitized_text,
                        response_source="conversational_llm",
                        response_type="therapeutic_reframe" if strategy == "cbt_support" else "conversational_dialogue",
                        model_name=provider_resp.model_name,
                        cbt_data=cbt_data_json,
                        llm_metadata={
                            "prompt_version": PROMPT_VERSION,
                            "provider": provider_resp.provider_name,
                            "model": provider_resp.model_name,
                            "strategy": strategy,
                            "fallback_used": False,
                            "latency_ms": provider_resp.latency_ms,
                            "prompt_tokens": provider_resp.prompt_tokens,
                            "completion_tokens": provider_resp.completion_tokens
                        }
                    )
                else:
                    logger.warning(f"LLM output validation failed: {failure_reason}. Executing fallback.")
                    fallback_reason = f"validation_failed ({failure_reason})"

            except Exception as e:
                logger.error(f"LLM Provider execution exception: {e}. Executing fallback.")
                fallback_reason = f"provider_error ({str(e)})"
        else:
            fallback_reason = "provider_unavailable"


        # 4. Fallback Execution Path
        fallback_content, fallback_cbt_data = fallback_service.generate_fallback(
            user_text=user_text,
            strategy=strategy,
            cbt_guidance=cbt_guidance,
            is_roman_urdu=is_roman_urdu
        )

        return LLMResult(
            content=fallback_content,
            response_source="conversational_llm_fallback",
            response_type="therapeutic_reframe" if strategy == "cbt_support" else "conversational_dialogue",
            model_name="template_fallback_v1",
            cbt_data=fallback_cbt_data,
            llm_metadata={
                "prompt_version": PROMPT_VERSION,
                "provider": "offline_fallback",
                "model": "template_fallback_v1",
                "strategy": strategy,
                "fallback_used": True,
                "fallback_reason": fallback_reason
            }
        )


llm_service = LLMService()
