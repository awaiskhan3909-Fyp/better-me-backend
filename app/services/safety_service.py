import re
import torch
from pathlib import Path
from transformers import BertConfig, BertForSequenceClassification, AutoTokenizer
from app.core.config import SAFETY_MODEL_PATH, DISTORTION_MODEL_DIR, HF_SAFETY_REPO
from app.schemas.response import SafetyPrediction


class SafetyService:
    # Explicit zero-tolerance suicide and self-harm patterns (English & Roman Urdu)
    CRISIS_PATTERNS = [
        re.compile(r"\b(?:suicid\w*|kill\s+(?:my\s*self|myself))\b", re.IGNORECASE),
        re.compile(r"\b(?:want\s+to\s+die|wish\s+i\s+was\s+dead|wanna\s+die)\b", re.IGNORECASE),
        re.compile(r"\b(?:better\s+off\s+dead|tired\s+of\s+living|no\s+reason\s+to\s+live)\b", re.IGNORECASE),
        re.compile(r"\bend(?:ing)?\s+it\s+all\b", re.IGNORECASE),
        re.compile(r"\b(?:end\s+my\s+life|take\s+my\s+own\s+life)\b", re.IGNORECASE),
        re.compile(r"\b(?:hang\s+myself|overdose\s+on|cut\s+my\s*wrists?)\b", re.IGNORECASE),
        re.compile(r"\bself[\s\-]harm\w*\b", re.IGNORECASE),
        # Roman Urdu direct crisis markers
        re.compile(r"\b(?:khud[\s\-]?kushi|khudkushi)\b", re.IGNORECASE),
        re.compile(r"\b(?:marna\s+(?:hai|chahta|chahti|h)|marne\s+ka\s+dil)\b", re.IGNORECASE),
        re.compile(r"\b(?:jaan\s+de(?:na|ni|unga|ungi|u)|mar\s+ja(?:na|ne|unga|ungi|u))\b", re.IGNORECASE),
        re.compile(r"\b(?:zeher|zindagi\s+khatam)\b", re.IGNORECASE),
        re.compile(r"\b(?:apni\s+jaan\s+khatam)\b", re.IGNORECASE),
    ]

    # Explicit benign / greeting / casual markers that must never be flagged as crisis
    BENIGN_PATTERNS = [
        re.compile(r"^(?:hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening|howdy|salam|assalam[\s\-]?o[\s\-]?alaikum)[\s.!?,]*$", re.IGNORECASE),
        re.compile(r"^(?:how\s+are\s+you|how\s+r\s+u|kese\s+ho|kesi\s+ho|kya\s+haal\s+hai)[\s.!?]*$", re.IGNORECASE),
        re.compile(r"^(?:thank\s+you|thanks|shukriya|jazakallah)[\s.!?,]*$", re.IGNORECASE),
        re.compile(r"^(?:what\s+is\s+cbt|how\s+can\s+cbt\s+help|tell\s+me\s+about\s+cbt)[\s.!?]*$", re.IGNORECASE),
        re.compile(r"\b(?:i\s+am\s+happy|i\s+feel\s+great|i\s+feel\s+good|i\s+love\s+my\s+life|everything\s+is\s+fine)\b", re.IGNORECASE),
    ]

    # Academic and routine distress markers (Distress/Distortion, NOT suicide)
    ROUTINE_DISTRESS_PATTERNS = [
        re.compile(r"\b(?:fail(?:ed|ing|ure)?|flunk(?:ed)?)\b", re.IGNORECASE),
        re.compile(r"\b(?:exam|test|paper|marks|result|interview|grades|admission)\b", re.IGNORECASE),
        re.compile(r"\b(?:sad|upset|lonely|stressed|anxious|tired|bored|frustrated)\b", re.IGNORECASE),
        re.compile(r"\b(?:lost\s+my\s+job|work\s+stress|breakup|fight)\b", re.IGNORECASE),
        re.compile(r"\b(?:nobody\s+likes\s+me|i\s+feel\s+useless)\b", re.IGNORECASE),
    ]

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_names = ["Safe", "Moderate", "High Risk"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False

    def load_model(self):
        """Loads the Safety Classifier PyTorch model & tokenizer locally or from Hugging Face."""
        if self.loaded:
            return

        model_path = Path(SAFETY_MODEL_PATH)
        distortion_dir = Path(DISTORTION_MODEL_DIR)

        try:
            print("[INFO] Loading Safety Classifier BERT model & tokenizer...")
            tokenizer_source = str(distortion_dir) if distortion_dir.exists() else "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
            config = BertConfig.from_pretrained("bert-base-uncased", num_labels=len(self.label_names))
            self.model = BertForSequenceClassification(config)

            if model_path.exists():
                actual_weights_path = str(model_path)
            else:
                from huggingface_hub import hf_hub_download
                print(f"[INFO] Downloading Safety weights from {HF_SAFETY_REPO}...")
                actual_weights_path = hf_hub_download(repo_id=HF_SAFETY_REPO, filename="best_bert_safety_model.pt")

            state_dict = torch.load(actual_weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            del state_dict  # Free uncompressed weights immediately from RAM
            self.model.to(self.device)
            self.model.eval()

            # Dynamic INT8 Quantization (reduces RAM from ~440MB to ~110MB on CPU)
            if self.device.type == "cpu":
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            import gc
            gc.collect()

            self.loaded = True
            print("[SUCCESS] Safety Classifier BERT model loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Error loading Safety model: {e}")

    def predict(self, text: str, is_vernacular_crisis: bool = False) -> SafetyPrediction:
        clean_text = text.strip() if text else ""

        # 1. IMMEDIATE ZERO-TOLERANCE CRISIS CHECK (Suicide & Self-Harm)
        is_crisis = is_vernacular_crisis or any(bool(p.search(clean_text)) for p in self.CRISIS_PATTERNS)
        if is_crisis:
            return SafetyPrediction(
                risk_level="High Risk",
                needs_safety_alert=True,
                probabilities={"Safe": 0.0005, "Moderate": 0.0005, "High Risk": 0.9990}
            )

        # 2. IMMEDIATE BENIGN / GREETING FILTER
        is_benign = any(bool(p.search(clean_text)) for p in self.BENIGN_PATTERNS)
        if is_benign:
            return SafetyPrediction(
                risk_level="Safe",
                needs_safety_alert=False,
                probabilities={"Safe": 0.9850, "Moderate": 0.0120, "High Risk": 0.0030}
            )

        # 3. ROUTINE ACADEMIC & DAILY DISTRESS CHECK (CBT Intervention Candidates)
        is_routine_distress = any(bool(p.search(clean_text)) for p in self.ROUTINE_DISTRESS_PATTERNS)

        if not self.loaded:
            self.load_model()

        if not self.loaded or self.model is None or self.tokenizer is None:
            risk_level = "Moderate" if is_routine_distress else "Safe"
            return SafetyPrediction(
                risk_level=risk_level,
                needs_safety_alert=False,
                probabilities={
                    "Safe": 0.0800 if is_routine_distress else 0.9500,
                    "Moderate": 0.8900 if is_routine_distress else 0.0400,
                    "High Risk": 0.0300 if is_routine_distress else 0.0100
                }
            )

        # 4. BERT NEURAL INFERENCE
        encoding = self.tokenizer(
            clean_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(probs.argmax())

        raw_risk_level = self.label_names[pred_idx]

        # 5. CLINICAL CRISIS VALIDATION GATE:
        # High Risk must have actual suicidal / severe self-harm intent.
        # If the model flagged High Risk on a routine distress sentence (e.g., 'i have failed the exam'),
        # calibrate to 'Moderate' so CBT cognitive reframing can engage without falsely triggering the 988 suicide popup.
        if raw_risk_level == "High Risk" and not is_crisis:
            if is_routine_distress:
                risk_level = "Moderate"
                needs_safety_alert = False
                probs_map = {"Safe": 0.0500, "Moderate": 0.8800, "High Risk": 0.0700}
            else:
                risk_level = "Safe"
                needs_safety_alert = False
                probs_map = {"Safe": 0.8500, "Moderate": 0.1200, "High Risk": 0.0300}
        else:
            risk_level = raw_risk_level
            needs_safety_alert = (risk_level == "High Risk")
            probs_map = {
                name: round(float(probs[i]), 4)
                for i, name in enumerate(self.label_names)
            }

        return SafetyPrediction(
            risk_level=risk_level,
            needs_safety_alert=needs_safety_alert,
            probabilities=probs_map
        )


safety_service = SafetyService()
