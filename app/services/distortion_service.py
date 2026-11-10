import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, BertForSequenceClassification
from app.core.config import DISTORTION_MODEL_DIR, HF_DISTORTION_REPO
from app.schemas.response import DistortionPrediction


CANONICAL_DISTORTIONS = {
    0: "All-or-Nothing Thinking",
    1: "Catastrophizing",
    2: "Emotional Reasoning",
    3: "Fortune Telling",
    4: "Mind Reading",
    5: "Overgeneralization"
}


class DistortionService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_map = CANONICAL_DISTORTIONS.copy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False

    def load_model(self):
        """Loads the Cognitive Distortion BERT model and tokenizer into memory once at startup."""
        if self.loaded:
            return

        import os
        model_dir = Path(DISTORTION_MODEL_DIR)
        env_dist_path = os.getenv("BERT_DISTORTION_PATH")
        drive_paths = [
            Path(env_dist_path) if env_dist_path else None,
            Path("/content/drive/MyDrive/bert_cognitive_distortions"),
            Path("/content/drive/MyDrive/bert_cognitive_distortions (Trained Model)"),
            Path("/content/drive/MyDrive/Models/Bert Model Training/bert_cognitive_distortions (Trained Model)"),
            Path("/content/drive/MyDrive/Models/bert_cognitive_distortions")
        ]
        
        found_drive_path = None
        for dp in drive_paths:
            if dp and dp.exists():
                found_drive_path = dp
                break

        if found_drive_path:
            print(f"📁 [DRIVE DETECTED] Found Cognitive Distortion model in Google Drive: {found_drive_path}")
            model_source = str(found_drive_path)
        elif model_dir.exists():
            model_source = str(model_dir)
        else:
            model_source = HF_DISTORTION_REPO

        try:
            print(f"[INFO] Loading Cognitive Distortion BERT model & tokenizer from: {model_source}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            self.model = BertForSequenceClassification.from_pretrained(model_source)
            self.model.to(self.device)
            self.model.eval()

            # Quantize dynamic to reduce RAM from 440MB to ~110MB on CPU
            if self.device.type == "cpu":
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            import gc
            gc.collect()

            # Always check label_map.json first to avoid generic HuggingFace LABEL_0...LABEL_5
            source_path = Path(model_source)
            label_map_path = source_path / "label_map.json" if source_path.exists() and source_path.is_dir() else (model_dir / "label_map.json" if model_dir.exists() else None)
            if label_map_path and label_map_path.exists():
                try:
                    with open(label_map_path, "r", encoding="utf-8") as f:
                        raw_map = json.load(f)
                        self.label_map = {int(k): v for k, v in raw_map.items()}
                except Exception:
                    pass

            # Only inspect config.id2label if it has genuine clinical distortion names (NOT LABEL_X)
            if (not self.label_map or any(str(v).startswith("LABEL_") for v in self.label_map.values())) and \
               hasattr(self.model, "config") and getattr(self.model.config, "id2label", None):
                candidate_map = {int(k): str(v) for k, v in self.model.config.id2label.items()}
                if not any(v.startswith("LABEL_") for v in candidate_map.values()):
                    self.label_map = candidate_map

            # If still unset or contains raw LABEL_, fallback to canonical CBT taxonomy
            if not self.label_map or any(str(v).startswith("LABEL_") for v in self.label_map.values()):
                self.label_map = CANONICAL_DISTORTIONS.copy()

            self.loaded = True
            print(f"[SUCCESS] Cognitive Distortion BERT model loaded! Active classes: {list(self.label_map.values())}")
        except Exception as e:
            print(f"[ERROR] Error loading Cognitive Distortion model: {e}")

    def predict(self, text: str) -> DistortionPrediction:
        if not self.loaded:
            self.load_model()

        if not self.loaded or self.model is None or self.tokenizer is None:
            return DistortionPrediction(
                predicted_class="None",
                confidence=0.0,
                all_probabilities={}
            )

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
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
            confidence = float(probs[pred_idx])

        # Enforce canonical distortion names: NEVER return raw LABEL_X
        predicted_class = self.label_map.get(pred_idx, CANONICAL_DISTORTIONS.get(pred_idx, "None"))
        if str(predicted_class).startswith("LABEL_"):
            predicted_class = CANONICAL_DISTORTIONS.get(pred_idx, "None")

        all_probs = {}
        for i, p in enumerate(probs):
            class_name = self.label_map.get(i, CANONICAL_DISTORTIONS.get(i, f"Distortion_{i}"))
            if str(class_name).startswith("LABEL_"):
                class_name = CANONICAL_DISTORTIONS.get(i, f"Distortion_{i}")
            all_probs[class_name] = round(float(p), 4)

        return DistortionPrediction(
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            all_probabilities=all_probs
        )


distortion_service = DistortionService()
