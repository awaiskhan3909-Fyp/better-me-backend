"""
==============================================================================
BETTER ME FYP: CLINICAL CBT LLM FINE-TUNING PIPELINE (GOOGLE COLAB T4 GPU)
==============================================================================
Model Picked: unsloth/llama-3-8b-Instruct-bnb-4bit
Alternative: unsloth/Llama-3.2-3B-Instruct-bnb-4bit (even faster) or unsloth/Qwen2.5-7B-Instruct-bnb-4bit

Hardware Compatibility:
- Tested for Google Colab Free Tier (Tesla T4 GPU with 15GB VRAM).
- Uses Unsloth 4-bit QLoRA: Uses only ~7.5 GB VRAM during training and ~5.2 GB during inference.
- Prevents CUDA Out-Of-Memory (OOM) errors and trains 2x faster.
==============================================================================
"""

import os
import sys
import json
import torch
from datasets import Dataset

# ==============================================================================
# 1. Configuration & Parameters
# ==============================================================================
# Choose the base model (unsloth optimized 4bit models)
MODEL_NAME = os.getenv("BASE_MODEL_NAME", "unsloth/llama-3-8b-Instruct-bnb-4bit")
MAX_SEQ_LENGTH = 2048
DTYPE = None  # None for auto-detection (Float16 or Bfloat16)
LOAD_IN_4BIT = True

OUTPUT_DIR = "better_me_cbt_llama3_lora"
HF_REPO_NAME = os.getenv("HF_EXPORT_REPO", "awaiskhan4039/better-me-cbt-llama3-lora")
HF_TOKEN = os.getenv("HF_TOKEN", None)


def main():
    print("\n" + "=" * 65)
    print("🚀 [Better Me FYP] Starting Clinical CBT LLM Fine-Tuning Pipeline...")
    print(f"📦 Base Model: {MODEL_NAME}")
    print(f"🖥️ GPU Available: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU FOUND'}")
    print("=" * 65 + "\n")

    if not torch.cuda.is_available():
        print("⚠️ WARNING: GPU not detected! Make sure you selected 'T4 GPU' in Colab Runtime Settings.")

    # ==========================================================================
    # 2. Load Model & Tokenizer via Unsloth
    # ==========================================================================
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("[ERROR] Unsloth is not installed. Please run:")
        print("pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"")
        print("pip install --no-deps trl peft accelerate bitsandbytes")
        sys.exit(1)

    print("[STEP 1/5] Loading 4-bit Quantized Base Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # ==========================================================================
    # 3. Add QLoRA Adapter Weights
    # ==========================================================================
    print("[STEP 2/5] Attaching Parameter-Efficient QLoRA Adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,  # Optimized to 0 for Unsloth speed
        bias="none",
        use_gradient_checkpointing="unsloth",  # Saves 70% memory
        random_state=42,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # ==========================================================================
    # 4. Prepare Dataset in Alpaca/Llama-3 Format
    # ==========================================================================
    print("[STEP 3/5] Loading and formatting CBT Clinical Dataset...")
    dataset_file = os.path.join(os.path.dirname(__file__), "cbt_dataset.json")
    if not os.path.exists(dataset_file):
        dataset_file = "cbt_dataset.json"

    with open(dataset_file, "r", encoding="utf-8") as f:
        cbt_raw_data = json.load(f)

    alpaca_prompt = """Below is an instruction that describes a clinical CBT therapeutic task, paired with an input containing the patient's distressed thought. Write a response that provides empathetic validation, Socratic cognitive restructuring, and a balanced reframe.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    raw_dataset = Dataset.from_list(cbt_raw_data)
    train_dataset = raw_dataset.map(formatting_prompts_func, batched=True)
    print(f"✅ Dataset prepared with {len(train_dataset)} clinical examples.")

    # ==========================================================================
    # 5. Execute Training via SFTTrainer
    # ==========================================================================
    print("[STEP 4/5] Initializing SFTTrainer and launching training...")
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,  # Short sequences
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=40,  # 40 steps for rapid convergence on Colab T4
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=OUTPUT_DIR,
        ),
    )

    trainer_stats = trainer.train()
    print(f"🎉 Training finished in {trainer_stats.metrics['train_runtime']:.2f} seconds!")

    # ==========================================================================
    # 6. Clinical Inference Test
    # ==========================================================================
    print("\n[STEP 5/5] Testing Fine-Tuned Model Generation...")
    FastLanguageModel.for_inference(model)

    test_input = "I didn't hear back from the company after my interview today. I'm definitely never getting a job and I'm a complete failure."
    test_instruction = "You are Better Me, a clinical CBT companion. The patient exhibits Catastrophizing and Overgeneralization. Validate, explore facts, and provide a balanced thought."

    prompt_formatted = alpaca_prompt.format(test_instruction, test_input, "")
    inputs = tokenizer([prompt_formatted], return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.7,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_part = generated_text.split("### Response:")[-1].strip()

    print("\n--- CLINICAL CBT MODEL RESPONSE ---")
    print(response_part)
    print("-----------------------------------\n")

    # ==========================================================================
    # 7. Save LoRA Adapters
    # ==========================================================================
    print(f"💾 Saving fine-tuned LoRA weights to '{OUTPUT_DIR}'...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Local model saved successfully to '{OUTPUT_DIR}'.")

    # Optional Push to Hugging Face Hub
    if HF_TOKEN:
        print(f"🚀 Pushing LoRA weights to Hugging Face Hub ({HF_REPO_NAME})...")
        try:
            model.push_to_hub(HF_REPO_NAME, token=HF_TOKEN)
            tokenizer.push_to_hub(HF_REPO_NAME, token=HF_TOKEN)
            print(f"🌟 Successfully pushed model to https://huggingface.co/{HF_REPO_NAME}!")
        except Exception as e:
            print(f"⚠️ Could not push to HF Hub: {e}")
    else:
        print("💡 TIP: Set HF_TOKEN environment variable to automatically push to Hugging Face Hub.")


if __name__ == "__main__":
    main()
