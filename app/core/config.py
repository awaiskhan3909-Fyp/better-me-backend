import os
from pathlib import Path
from dotenv import load_dotenv

# Paths configuration
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Load environment variables from .env
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

LOCAL_MODELS_DIR = PROJECT_ROOT / "Models"

DISTORTION_MODEL_DIR = LOCAL_MODELS_DIR / "Bert Model Training" / "bert_cognitive_distortions (Trained Model)"
SAFETY_MODEL_PATH = LOCAL_MODELS_DIR / "Safety Classifier Model" / "bert_safety_classifier (Trained Model Files)" / "best_bert_safety_model.pt"

# Resolve CBT JSON Templates directory (check local backend bundle, then workspace root)
CBT_TEMPLATES_DIR = BACKEND_DIR / "CBT JSON Templates"
if not CBT_TEMPLATES_DIR.exists():
    CBT_TEMPLATES_DIR = PROJECT_ROOT / "CBT JSON Templates"
if not CBT_TEMPLATES_DIR.exists():
    CBT_TEMPLATES_DIR = PROJECT_ROOT / "CBT Jason Templates"

# Hugging Face Repositories for Cloud Hosting (BERT Classifiers)
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_DISTORTION_REPO = os.getenv("HF_DISTORTION_REPO", "awaiskhan4039/bert-cognitive-distortions")
HF_SAFETY_REPO = os.getenv("HF_SAFETY_REPO", "awaiskhan4039/bert-safety-classifier")

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))
APP_ENV = os.getenv("APP_ENV", "development")
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
)
ALLOWED_ORIGINS = [origin.strip().strip("'\"") for origin in _raw_origins.split(",") if origin.strip()]
ALLOW_ORIGIN_REGEX = os.getenv("ALLOW_ORIGIN_REGEX", r"^https:\/\/.*\.vercel\.app$|^https:\/\/.*\.ngrok-free\.app$")

# Database Configuration (Defaults to local SQLite, environment variable override for PostgreSQL)
DEFAULT_SQLITE_PATH = BACKEND_DIR / "better_me.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")

# Conversation Intelligence Configuration
CONVERSATION_HISTORY_LIMIT = int(os.getenv("CONVERSATION_HISTORY_LIMIT", 10))

# Conversational LLM Configuration (Google Gemini API)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
LLM_FALLBACK_ENABLED = os.getenv("LLM_FALLBACK_ENABLED", "true").lower() == "true"


