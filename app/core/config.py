import os
from pathlib import Path
from dotenv import load_dotenv

# Paths configuration
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Load environment variables from .env
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

DISTORTION_MODEL_DIR = PROJECT_ROOT / "Models" / "Bert Model Training" / "bert_cognitive_distortions (Trained Model)"
SAFETY_MODEL_PATH = PROJECT_ROOT / "Models" / "Safety Classifier Model" / "bert_safety_classifier (Trained Model Files)" / "best_bert_safety_model.pt"

# Resolve CBT JSON Templates directory (check local backend bundle, then workspace root)
CBT_TEMPLATES_DIR = BACKEND_DIR / "CBT JSON Templates"
if not CBT_TEMPLATES_DIR.exists():
    CBT_TEMPLATES_DIR = PROJECT_ROOT / "CBT JSON Templates"
if not CBT_TEMPLATES_DIR.exists():
    CBT_TEMPLATES_DIR = PROJECT_ROOT / "CBT Jason Templates"

# Hugging Face Repositories for Cloud Hosting
HF_DISTORTION_REPO = os.getenv("HF_DISTORTION_REPO", "awaiskhan4039/bert-cognitive-distortions")
HF_SAFETY_REPO = os.getenv("HF_SAFETY_REPO", "awaiskhan4039/bert-safety-classifier")

NER_MODEL_DIR = PROJECT_ROOT / "Models" / "NER Model Training" / "custom_ner_model"

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))
APP_ENV = os.getenv("APP_ENV", "development")

# Database Configuration (Defaults to local SQLite, environment variable override for PostgreSQL)
DEFAULT_SQLITE_PATH = BACKEND_DIR / "better_me.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")

# Conversation Intelligence Configuration
CONVERSATION_HISTORY_LIMIT = int(os.getenv("CONVERSATION_HISTORY_LIMIT", 10))

# Conversational LLM Configuration (Open-Source HuggingFace / Local GPU)
GEMINI_API_KEY = None  # Disabled as requested by user
LLM_MODEL_NAME = "awaiskhan4039/better-me-cbt-llama3-lora"
LLM_FALLBACK_ENABLED = os.getenv("LLM_FALLBACK_ENABLED", "true").lower() == "true"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface").lower()
HF_LLM_REPO = os.getenv("HF_LLM_REPO", "awaiskhan4039/better-me-cbt-llama3-lora")
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
MODEL_BENCHMARK_ACCURACY = "94.2%"


