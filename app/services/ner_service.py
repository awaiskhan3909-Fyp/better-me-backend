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
                            end=match.end()
                        )
                    )

        # Sort extracted entities by start position
        results.sort(key=lambda x: x.start if x.start is not None else 0)
        return results


ner_service = NERService()
