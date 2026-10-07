import torch
from pathlib import Path
from transformers import BertConfig, BertForSequenceClassification, AutoTokenizer
from app.core.config import SAFETY_MODEL_PATH, DISTORTION_MODEL_DIR, HF_SAFETY_REPO
from app.schemas.response import SafetyPrediction


class SafetyService:
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

    def predict(self, text: str) -> SafetyPrediction:
        if not self.loaded:
            self.load_model()

        if not self.loaded or self.model is None or self.tokenizer is None:
            return SafetyPrediction(
                risk_level="Safe",
                needs_safety_alert=False,
                probabilities={"Safe": 1.0, "Moderate": 0.0, "High Risk": 0.0}
            )

        encoding = self.tokenizer(
            text,
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

        risk_level = self.label_names[pred_idx]
        needs_safety_alert = (risk_level == "High Risk")

        all_probs = {
            name: round(float(probs[i]), 4)
            for i, name in enumerate(self.label_names)
        }

        return SafetyPrediction(
            risk_level=risk_level,
            needs_safety_alert=needs_safety_alert,
            probabilities=all_probs
        )


safety_service = SafetyService()
