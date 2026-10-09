---
title: Better Me Backend
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
---

# 🧠 Better Me AI Backend API

FastAPI Cognitive Behavioral Therapy (CBT) Backend, Clinical Memory Engine, and Hugging Face Llama-3-8B Inference Service.

> **BS Final Year Project (FYP)**  
> **Supervisor:** Mam Farnaz Akbar  
> **Team Members:** Awais Khan, Saad Abdullah, Ajiya Asif  
> **Master Documentation:** Refer to [`../README.md`](../README.md) for full project architecture and clinical justifications.

---

## 📌 Features & Modules

1. **Longitudinal Clinical Memory Engine (`app/db/phase1_clinical_memory_schema.sql`):**
   - `patient_clinical_profiles`: Stores baseline PHQ-9/GAD-7 intake, persistent themes, core cognitive schemas.
   - `episodic_therapy_memories`: Multi-session milestone tracking and semantic vector retrieval.

2. **Aaron Beck's 5-Column Thought Records (`app/api/endpoints/thought_records.py`):**
   - CRUD and cognitive distortion classification for patient automatic thoughts.
   - Evidence-based reframing and distress reduction delta analytics.

3. **Llama-3-8B QLoRA Inference Client (`app/services/llm/huggingface_provider.py`):**
   - Fine-tuned adapter: [`awaiskhan4039/better-me-cbt-llama3-lora`](https://huggingface.co/awaiskhan4039/better-me-cbt-llama3-lora)
   - Socratic guidance prompt assembly and clinical crisis safety triage.

4. **Safety & Crisis Triage (`app/api/endpoints/analyze.py`):**
   - BERT classification for immediate self-harm/suicide crisis override (988 protocol).

---

## 🚀 Quick Start

```bash
# 1. Virtual Environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows

# 2. Dependencies
pip install -r requirements.txt

# 3. Start Server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Swagger API Docs: `http://localhost:8000/docs`
