import json
import re
from typing import Dict, Any, List, Optional
from app.core.config import CBT_TEMPLATES_DIR
from app.schemas.response import CBTGuidance


class CBTService:
    def __init__(self):
        self.templates_db: Dict[str, Dict[str, Any]] = {}
        self.loaded = False
        self.load_templates()

    def load_templates(self):
        """Load CBT JSON template files into memory from CBT_TEMPLATES_DIR."""
        if self.loaded:
            return

        file_mapping = {
            "Catastrophizing": "cbt_catastrophizing_templates.json",
            "Mind Reading": "cbt_mind_reading_templates.json",
            "Overgeneralization": "cbt_overgeneralization_templates.json",
            "All-or-Nothing Thinking": "cbt_all_or_nothing_thinking_templates.json",
            "Emotional Reasoning": "cbt_emotional_reasoning_templates.json",
            "Fortune Telling": "cbt_fortune_telling_templates.json",
        }

        print("[INFO] Loading CBT JSON templates...")
        for distortion, filename in file_mapping.items():
            file_path = CBT_TEMPLATES_DIR / filename
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        self.templates_db[distortion] = json.load(f)
                    print(f"  [LOADED] {distortion} ({len(self.templates_db[distortion].get('templates', []))} templates)")
                except Exception as e:
                    print(f"  [ERROR] Failed to load {filename}: {e}")
            else:
                print(f"  [WARNING] Template file not found: {file_path}")

        self.loaded = True

    def get_cbt_guidance(self, user_text: str, distortion_name: str) -> CBTGuidance:
        """
        Deterministically selects a matching CBT template from existing JSON files
        based on the predicted cognitive distortion and user input context.
        """
        if not self.loaded:
            self.load_templates()

        text_lower = user_text.lower()

        # If distortion exists in JSON templates database
        if distortion_name in self.templates_db:
            distortion_data = self.templates_db[distortion_name]
            templates_list = distortion_data.get("templates", [])

            if templates_list:
                selected_template = self._match_template(text_lower, templates_list)
                return CBTGuidance(
                    detected_distortion=distortion_name,
                    template_id=selected_template.get("id"),
                    title=selected_template.get("title"),
                    explanation=selected_template.get("explanation"),
                    reframing_question=selected_template.get("reflection_question"),
                    balanced_thought_guidance=selected_template.get("balanced_reframe", ""),
                    small_action=selected_template.get("behavioral_action")
                )

        # Fallback if prediction is None or distortion template unavailable
        return CBTGuidance(
            detected_distortion=distortion_name or "General",
            template_id=0,
            title="General Reflection",
            explanation="Recognizing our thought patterns allows us to evaluate them objectively rather than accepting them as facts.",
            reframing_question="What evidence supports or challenges this thought right now?",
            balanced_thought_guidance="Thank you for sharing your thoughts. Exploring alternate possibilities can help restore balance.",
            small_action="Take 3 deep, grounding breaths and write down one objective fact about your situation."
        )

    def _match_template(self, text_lower: str, templates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Deterministic matching score based on keyword overlap between user text
        and template titles/explanations.
        """
        best_score = -1
        best_template = templates[0]

        for t in templates:
            score = 0
            # Check title words
            title_words = re.findall(r'\w+', t.get("title", "").lower())
            for word in title_words:
                if len(word) > 3 and word in text_lower:
                    score += 2

            # Check explanation words
            exp_words = re.findall(r'\w+', t.get("explanation", "").lower())
            for word in exp_words:
                if len(word) > 4 and word in text_lower:
                    score += 1

            if score > best_score:
                best_score = score
                best_template = t

        return best_template


cbt_service = CBTService()
