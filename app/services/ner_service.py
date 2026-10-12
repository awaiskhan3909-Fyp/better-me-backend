import re
from typing import List
from app.schemas.response import EntityItem


class NERService:
    def __init__(self):
        # Entity categories & terms: RELATIONSHIP, WORKPLACE, FAMILY, HEALTH, ACADEMIC, FINANCIAL, SOCIAL
        self.entity_keywords = {
            "RELATIONSHIP": [
                "boyfriend", "girlfriend", "partner", "spouse", "husband", "wife",
                "fiancé", "fiancée", "date", "dating", "relationship", "breakup", "divorce", "ex"
            ],
            "WORKPLACE": [
                "boss", "manager", "supervisor", "office", "job", "work", "workplace",
                "colleague", "coworker", "employment", "career", "company", "promotion",
                "interview", "client", "workload"
            ],
            "FAMILY": [
                "mother", "father", "parents", "brother", "sister", "mom", "dad",
                "family", "son", "daughter", "grandmother", "grandfather", "sibling",
                "child", "kids", "cousin", "relative"
            ],
            "HEALTH": [
                "anxiety", "depression", "panic attack", "chest pain", "insomnia",
                "fatigue", "headache", "migraine", "mental health", "stress", "illness",
                "therapy", "therapist", "medication", "exhaustion", "burnout"
            ],
            "ACADEMIC": [
                "exam", "exams", "midterm", "finals", "quiz", "test", "tests",
                "gpa", "grade", "grades", "university", "college", "school",
                "professor", "teacher", "assignment", "homework", "thesis",
                "degree", "coursework", "semester", "scholarship", "lecture"
            ],
            "FINANCIAL": [
                "money", "rent", "debt", "bills", "bill", "broke", "loan", "loans",
                "expenses", "savings", "salary", "budget", "afford", "credit card",
                "bank account", "tuition", "financial"
            ],
            "SOCIAL": [
                "friend", "friends", "best friend", "close friend", "friendship",
                "friend group", "peers", "peer pressure", "party", "parties",
                "gathering", "hangout", "social media", "isolation", "isolated",
                "lonely", "loneliness", "left out", "excluded"
            ]
        }

    def extract_entities(self, text: str) -> List[EntityItem]:
        """
        Extracts emotional context entities (RELATIONSHIP, WORKPLACE, FAMILY, HEALTH, ACADEMIC, FINANCIAL, SOCIAL)
        from user statement based on the project's expanded NER specification.
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
                            end=match.end(),
                            confidence=0.975
                        )
                    )

        # Sort extracted entities by start position
        results.sort(key=lambda x: x.start if x.start is not None else 0)

        # Filter out casual slang vocatives (e.g., 'who are you brother', 'kon ho ap bhai', 'thanks bro')
        text_lower = text.lower()
        is_casual_inquiry = bool(re.search(r"\b(?:who|what|hi|hello|hey|thanks|thank|kon|kaun|ap|kya|bhai|bro)\b", text_lower))
        
        filtered_results: List[EntityItem] = []
        for ent in results:
            if ent.label == "FAMILY" and ent.text.lower() in ["brother", "bro"]:
                has_family_possession = bool(re.search(r"\b(?:my|our|his|her|their|with|and)\s+(?:brother|bro)\b", text_lower))
                if is_casual_inquiry and not has_family_possession and len(text.split()) <= 6:
                    continue
            filtered_results.append(ent)

        return filtered_results


ner_service = NERService()
