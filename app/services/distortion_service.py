import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, BertForSequenceClassification
from app.core.config import DISTORTION_MODEL_DIR
from app.schemas.response import DistortionPrediction


class DistortionService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_map = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False

    def load_model(self):
        """Loads the Cognitive Distortion BERT model and tokenizer into memory once at startup."""
        if self.loaded:
            return

        model_dir = Path(DISTORTION_MODEL_DIR)
        model_source = str(model_dir) if model_dir.exists() else HF_DISTORTION_REPO

        try:
            print(f"[INFO] Loading Cognitive Distortion BERT model & tokenizer from: {model_source}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            self.model = BertForSequenceClassification.from_pretrained(model_source)
            self.model.to(self.device)
            self.model.eval()

            label_map_path = model_dir / "label_map.json" if model_dir.exists() else None
            if label_map_path and label_map_path.exists():
                with open(label_map_path, "r", encoding="utf-8") as f:
                    raw_map = json.load(f)
                    self.label_map = {int(k): v for k, v in raw_map.items()}
            else:
                try:
                    from huggingface_hub import hf_hub_download
                    downloaded_label = hf_hub_download(repo_id=HF_DISTORTION_REPO, filename="label_map.json")
                    with open(downloaded_label, "r", encoding="utf-8") as f:
                        raw_map = json.load(f)
                        self.label_map = {int(k): v for k, v in raw_map.items()}
                except Exception:
                    self.label_map = {
                        0: "All-or-Nothing Thinking",
                        1: "Catastrophizing",
                        2: "Emotional Reasoning",
                        3: "Fortune Telling",
                        4: "Mind Reading",
                        5: "Overgeneralization"
                    }

            self.loaded = True
            print("[SUCCESS] Cognitive Distortion BERT model loaded successfully!")
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

        predicted_class = self.label_map.get(pred_idx, "Unknown")
        all_probs = {
            self.label_map.get(i, f"Class_{i}"): round(float(p), 4)
            for i, p in enumerate(probs)
        }

        return DistortionPrediction(
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            all_probabilities=all_probs
        )


distortion_service = DistortionService()
