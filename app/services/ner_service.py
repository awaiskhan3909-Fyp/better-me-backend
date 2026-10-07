import re
from typing import List
from app.schemas.response import EntityItem


class NERService:
    def __init__(self):
        # Entity categories & terms from custom_ner_emotional_text.py
        self.entity_keywords = {
            "RELATIONSHIP": [
                "boyfriend", "girlfriend", "partner", "spouse", "husband", "wife",
                "friend", "close friend", "best friend", "relationship", "breakup", "ex"
            ],
            "WORKPLACE": [
                "boss", "manager", "office", "job", "work", "workplace", "colleague",
                "coworker", "employment", "career", "company"
            ],
            "FAMILY": [
                "mother", "father", "parents", "brother", "sister", "mom", "dad",
                "family", "son", "daughter", "grandmother", "grandfather"
            ],
            "HEALTH": [
                "anxiety", "depression", "panic attack", "chest pain", "insomnia",
                "fatigue", "headache", "mental health", "stress", "illness"
            ]
        }

    def extract_entities(self, text: str) -> List[EntityItem]:
        """
        Extracts emotional context entities (RELATIONSHIP, WORKPLACE, FAMILY, HEALTH)
        from user statement based on the project's NER specification.
        """
        results: List[EntityItem] = []

        for label, keywords in self.entity_keywords.items():
            for kw in keywords:
                pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
                for match in pattern.finditer(text):
                    results.append(
                        EntityItem(
                            text=match.group(0),
                            label=label,
                            start=match.start(),
                            end=match.end()
                        )
                    )

        # Sort extracted entities by start position
        results.sort(key=lambda x: x.start if x.start is not None else 0)
        return results


ner_service = NERService()
