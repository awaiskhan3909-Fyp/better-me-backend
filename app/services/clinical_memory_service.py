import re
import uuid
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from datetime import datetime, timezone

from app.db.models import (
    User,
    Conversation,
    Message,
    CBTThoughtRecord,
    PatientClinicalProfile,
    EpisodicTherapyMemory,
    UserIntakeAssessment
)


class ClinicalMemoryService:
    """
    Longitudinal Clinical & Cross-Session Context Memory Service.
    Maintains persistent patient profiles, recurring distortion habits,
    past breakthrough episodic memories, thought records, behavioral homework,
    and cross-session chat recall across all conversations for an account.
    """

    TRIGGER_KEYWORDS = {
        "Academic & Exams": ["exam", "test", "quiz", "grade", "gpa", "study", "studying", "marks", "fail", "failed", "assignment", "deadline", "university", "college", "fyp", "parhai"],
        "Career & Workplace": ["job", "interview", "boss", "work", "office", "career", "promotion", "resume", "colleague", "manager", "salary", "naukri"],
        "Social & Relationships": ["friend", "friends", "partner", "relationship", "breakup", "lonely", "rejected", "rejection", "ignoring", "family", "parents", "brother", "sister", "sibling", "siblings", "mother", "father", "mom", "dad", "dost"],
        "Health & Somatic": ["sleep", "insomnia", "tired", "exhausted", "headache", "panic", "heartbeat", "health", "sick", "neend"],
        "Self-Worth & Performance": ["failure", "useless", "worthless", "mistake", "not good enough", "loser", "perfectionist"]
    }

    STOPWORDS = {
        # English
        "the", "and", "for", "that", "this", "with", "from", "have", "what", "when", "where",
        "which", "about", "would", "could", "should", "there", "their", "been", "were", "will",
        "your", "you", "they", "them", "just", "like", "know", "really", "very", "more", "some",
        "than", "then", "into", "also", "feel", "feeling", "felt", "today", "still", "tell",
        # Roman Urdu
        "hai", "hain", "tha", "thi", "the", "kya", "kiya", "mein", "main", "mera", "meri", "mere",
        "aap", "tum", "hum", "woh", "yeh", "aur", "par", "nahi", "nhi", "kuch", "bhi", "kar",
        "karna", "raha", "rahi", "rahe", "ho", "hota", "hoti", "mujhe", "mujy", "mujhy", "apni",
        "apna", "apne", "liye", "sath", "saath", "baat", "bohot", "bahut", "lekin", "magar", "toh"
    }

    PAST_RECALL_PATTERNS = [
        re.compile(r"\b(?:remember|recall|last\s+time|previous(?:ly)?|earlier|before|past\s+chat|last\s+session|previous\s+session|other\s+chat)\b", re.IGNORECASE),
        re.compile(r"\b(?:what\s+did\s+(?:i|we)\s+(?:talk|discuss|say)|did\s+i\s+tell\s+you|as\s+i\s+mentioned|my\s+homework|follow[\s\-]?up)\b", re.IGNORECASE),
        re.compile(r"\b(?:yaad|pichl[aei]|pehl[aei]|guzashta|bataya\s+tha|kaha\s+tha|baat\s+hui\s+thi|baat\s+ki\s+thi|purani\s+chat|kal\s+(?:ki|wali)\s+baat)\b", re.IGNORECASE),
    ]

    def is_past_memory_query(self, text: str) -> bool:
        if not text:
            return False
        return any(bool(p.search(text)) for p in self.PAST_RECALL_PATTERNS)

    def _extract_keywords(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z0-9']+", (text or "").lower())
        return [t for t in tokens if len(t) >= 3 and t not in self.STOPWORDS]

    def get_or_create_profile(self, user_id: uuid.UUID, db: Session) -> PatientClinicalProfile:
        profile = db.query(PatientClinicalProfile).filter(PatientClinicalProfile.user_id == user_id).first()
        if not profile:
            intake = db.query(UserIntakeAssessment).filter(UserIntakeAssessment.user_id == user_id).first()
            initial_triggers = list(intake.primary_focus) if intake and intake.primary_focus else []
            initial_distortions = {d: 1 for d in intake.familiar_distortions} if intake and intake.familiar_distortions else {}

            profile = PatientClinicalProfile(
                user_id=user_id,
                primary_triggers=initial_triggers,
                dominant_distortions=initial_distortions,
                core_beliefs=[],
                effective_reframes=[],
                active_homework=None,
                last_session_summary=None,
                total_sessions_completed=0
            )
            db.add(profile)
            db.commit()
            db.refresh(profile)

        return profile

    def record_turn_insight(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        user_text: str,
        detected_distortion: str,
        reframe_text: str,
        db: Session
    ):
        """
        Incrementally updates patient's clinical memory after an exchange.
        """
        profile = self.get_or_create_profile(user_id, db)

        # 1. Update Dominant Distortions Count
        if detected_distortion and detected_distortion not in ["None", "Normal", "no distortion", "Safety Intervention"]:
            dist_map = dict(profile.dominant_distortions or {})
            dist_map[detected_distortion] = dist_map.get(detected_distortion, 0) + 1
            profile.dominant_distortions = dist_map
            flag_modified(profile, "dominant_distortions")

        # 2. Extract and Update Triggers
        text_lower = user_text.lower()
        triggers = list(profile.primary_triggers or [])
        triggers_changed = False
        for category, kws in self.TRIGGER_KEYWORDS.items():
            if any(re.search(rf"\b{kw}\b", text_lower) for kw in kws):
                if category not in triggers:
                    triggers.append(category)
                    triggers_changed = True
        if triggers_changed:
            profile.primary_triggers = triggers
            flag_modified(profile, "primary_triggers")

        # 3. Maintain rolling last-topic summary even for non-distortion turns
        if len(user_text.strip()) >= 20:
            profile.last_session_summary = user_text.strip()[:220]

        # 4. Log Episodic Therapy Memory if significant distortion reframed
        if detected_distortion and detected_distortion not in ["None", "Normal", "no distortion", "Safety Intervention"]:
            recent_ep = (
                db.query(EpisodicTherapyMemory)
                .filter(
                    EpisodicTherapyMemory.conversation_id == conversation_id,
                    EpisodicTherapyMemory.distortion_type == detected_distortion
                )
                .first()
            )

            if not recent_ep and len(user_text) >= 15:
                episode = EpisodicTherapyMemory(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    situation_context=user_text[:220],
                    distorted_thought=user_text[:220],
                    distortion_type=detected_distortion,
                    rational_reframe=reframe_text[:280],
                    breakthrough_notes=f"Reframed {detected_distortion} with Socratic questioning."
                )
                db.add(episode)

        profile.updated_at = datetime.now(timezone.utc)
        db.commit()

    def _retrieve_cross_session_exchanges(
        self,
        user_id: uuid.UUID,
        current_text: str,
        db: Session,
        current_conversation_id: Optional[uuid.UUID] = None,
        max_exchanges: int = 4
    ) -> Tuple[List[Conversation], List[Dict[str, str]]]:
        """
        Retrieves past sessions and scores past (user, ai) chat exchanges across
        previous conversations belonging to this user so the model can recall and
        answer questions related to earlier chats.
        """
        past_convs_query = db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.is_archived == False
        )
        if current_conversation_id is not None:
            past_convs_query = past_convs_query.filter(Conversation.id != current_conversation_id)

        past_convs = (
            past_convs_query
            .order_by(Conversation.updated_at.desc())
            .limit(15)
            .all()
        )
        if not past_convs:
            return [], []

        conv_map = {c.id: c for c in past_convs}
        conv_ids = list(conv_map.keys())

        # Fetch recent messages from those past conversations
        past_messages = (
            db.query(Message)
            .filter(Message.conversation_id.in_(conv_ids))
            .order_by(Message.conversation_id, Message.sequence_number.asc())
            .all()
        )

        # Group messages into (user -> next ai) exchanges per conversation
        by_conv: Dict[uuid.UUID, List[Message]] = {}
        for m in past_messages:
            by_conv.setdefault(m.conversation_id, []).append(m)

        query_keywords = set(self._extract_keywords(current_text))
        explicit_recall = self.is_past_memory_query(current_text)

        scored_exchanges: List[Tuple[float, datetime, Dict[str, str]]] = []
        for conv_idx, conv in enumerate(past_convs):
            msgs = by_conv.get(conv.id, [])
            if not msgs:
                continue

            title_tokens = set(self._extract_keywords(conv.title or ""))
            date_str = conv.updated_at.strftime("%b %d") if conv.updated_at else "Past"

            i = 0
            while i < len(msgs):
                if msgs[i].sender_type == "user":
                    u_text = msgs[i].content.strip()
                    a_text = ""
                    if i + 1 < len(msgs) and msgs[i + 1].sender_type == "ai":
                        a_text = msgs[i + 1].content.strip()
                        i += 2
                    else:
                        i += 1

                    # Skip trivial 1-word greetings like "hi" / "ok"
                    if len(u_text) < 8:
                        continue

                    combined_tokens = set(self._extract_keywords(u_text)) | set(self._extract_keywords(a_text)) | title_tokens
                    overlap = query_keywords & combined_tokens

                    # Prefix/substring boost for inflected words (e.g. exam/exams, fail/failed)
                    partial_hits = 0
                    if query_keywords:
                        combined_lower = f"{conv.title or ''} {u_text} {a_text}".lower()
                        for qk in query_keywords:
                            if len(qk) >= 4 and qk not in overlap and qk in combined_lower:
                                partial_hits += 1

                    relevance_score = (len(overlap) * 3.0) + (partial_hits * 1.5)
                    recency_bonus = max(0.0, 1.5 - (conv_idx * 0.15))

                    if relevance_score > 0 or explicit_recall:
                        scored_exchanges.append((
                            relevance_score + recency_bonus,
                            msgs[min(i - 1, len(msgs) - 1)].created_at or conv.updated_at,
                            {
                                "session_title": conv.title or "Past Session",
                                "date": date_str,
                                "user": u_text[:220],
                                "ai": a_text[:220],
                                "matched": str(relevance_score > 0),
                            }
                        ))
                else:
                    i += 1

        # Sort by relevance + recency descending
        scored_exchanges.sort(key=lambda item: (item[0], item[1]), reverse=True)

        selected: List[Dict[str, str]] = []
        for _, _, ex in scored_exchanges:
            selected.append(ex)
            if len(selected) >= max_exchanges:
                break

        # If nothing matched keywords AND not an explicit recall phrase, still include up to 2 most recent
        # meaningful past exchanges so the model always has light cross-session awareness
        if not selected:
            for conv in past_convs[:3]:
                msgs = by_conv.get(conv.id, [])
                date_str = conv.updated_at.strftime("%b %d") if conv.updated_at else "Past"
                for idx in range(len(msgs) - 1, -1, -1):
                    if msgs[idx].sender_type == "user" and len(msgs[idx].content.strip()) >= 12:
                        ai_reply = msgs[idx + 1].content.strip()[:200] if (idx + 1 < len(msgs) and msgs[idx + 1].sender_type == "ai") else ""
                        selected.append({
                            "session_title": conv.title or "Past Session",
                            "date": date_str,
                            "user": msgs[idx].content.strip()[:200],
                            "ai": ai_reply,
                            "matched": "False",
                        })
                        break
                if len(selected) >= 2:
                    break

        return past_convs[:6], selected

    def build_clinical_memory_briefing(
        self,
        user_id: Optional[uuid.UUID],
        current_text: str,
        db: Session,
        current_conversation_id: Optional[uuid.UUID] = None
    ) -> str:
        """
        Builds a rich, compact longitudinal clinical & cross-session memory block for the LLM.
        """
        if not user_id:
            return ""

        user = db.query(User).filter(User.id == user_id).first()
        profile = db.query(PatientClinicalProfile).filter(PatientClinicalProfile.user_id == user_id).first()
        intake = db.query(UserIntakeAssessment).filter(UserIntakeAssessment.user_id == user_id).first()

        lines = ["--- LONGITUDINAL PATIENT CLINICAL MEMORY & PAST CHAT CONTEXT ---"]

        if user and user.full_name:
            lines.append(f"Patient Name: {user.full_name}")

        if intake:
            if intake.primary_goal:
                lines.append(f"Patient Primary Goal: {intake.primary_goal}")
            if intake.primary_focus:
                lines.append(f"Focus Areas: {', '.join(intake.primary_focus)}")

        if profile:
            if profile.primary_triggers:
                lines.append(f"Recurring Triggers: {', '.join(profile.primary_triggers)}")
            if profile.dominant_distortions:
                sorted_dist = sorted(profile.dominant_distortions.items(), key=lambda x: x[1], reverse=True)[:3]
                dist_str = ", ".join([f"{k} ({v}x)" for k, v in sorted_dist])
                lines.append(f"Dominant Historical Distortions: {dist_str}")
            if profile.active_homework:
                lines.append(f"Active Behavioral Homework: {profile.active_homework}")
            if profile.last_session_summary:
                lines.append(f"Recent Topic Summary: {profile.last_session_summary[:180]}")

        # 1. Cross-Session Past Conversations & Relevant Chat Excerpts
        past_convs, past_exchanges = self._retrieve_cross_session_exchanges(
            user_id=user_id,
            current_text=current_text,
            db=db,
            current_conversation_id=current_conversation_id,
            max_exchanges=4
        )

        if past_convs:
            session_summaries = [
                f"{c.title or 'Session'} ({c.updated_at.strftime('%b %d') if c.updated_at else 'Past'})"
                for c in past_convs[:5]
            ]
            lines.append(f"Previous Therapy Sessions: {' | '.join(session_summaries)}")

        if past_exchanges:
            lines.append("Relevant Past Chat Excerpts (From Earlier Sessions):")
            for ex in past_exchanges:
                lines.append(f"  - [{ex['date']} | {ex['session_title']}] User said: \"{ex['user']}\"")
                if ex["ai"]:
                    lines.append(f"    Assistant replied: \"{ex['ai']}\"")

        # 2. Thought Records & Episodic Breakthroughs
        query_keywords = set(self._extract_keywords(current_text))
        thought_records = (
            db.query(CBTThoughtRecord)
            .filter(CBTThoughtRecord.user_id == user_id)
            .order_by(CBTThoughtRecord.created_at.desc())
            .limit(8)
            .all()
        )
        if thought_records:
            ranked_tr = sorted(
                thought_records,
                key=lambda tr: len(query_keywords & set(self._extract_keywords(f"{tr.situation} {tr.automatic_thought} {tr.balanced_thought}"))),
                reverse=True
            )[:2]
            lines.append("Past Thought Records:")
            for tr in ranked_tr:
                lines.append(
                    f"  - Situation: \"{tr.situation[:120]}\" | Thought: \"{tr.automatic_thought[:120]}\" "
                    f"| Balanced Reframe: \"{tr.balanced_thought[:140]}\""
                )

        episodes = (
            db.query(EpisodicTherapyMemory)
            .filter(EpisodicTherapyMemory.user_id == user_id)
            .order_by(EpisodicTherapyMemory.created_at.desc())
            .limit(8)
            .all()
        )
        if episodes:
            ranked_eps = sorted(
                episodes,
                key=lambda ep: len(query_keywords & set(self._extract_keywords(f"{ep.situation_context} {ep.distorted_thought} {ep.rational_reframe}"))),
                reverse=True
            )[:2]
            lines.append("Key Past Therapy Breakthroughs:")
            for ep in ranked_eps:
                lines.append(f"  - [{ep.distortion_type}] \"{ep.situation_context[:120]}\" -> Reframe: \"{ep.rational_reframe[:140]}\"")

        if self.is_past_memory_query(current_text):
            lines.append(
                "MEMORY RECALL ALERT: The user is explicitly referring to or asking about a past conversation/session. "
                "Directly use the Previous Therapy Sessions, Past Chat Excerpts, Thought Records, and Active Homework above "
                "to show accurate continuity and answer their question clearly. Do NOT invent memories that are not listed above."
            )
        else:
            lines.append(
                "CLINICAL DIRECTIVE: Use this cross-session history naturally whenever the user's current message relates to "
                "earlier topics, triggers, or coping strategies. NEVER invent details not present above."
            )
        lines.append("LANGUAGE NOTE: Past notes may be in English or Roman Urdu; ALWAYS reply strictly in the language of the current user message.")
        lines.append("----------------------------------------------------------------")
        return "\n".join(lines)


clinical_memory_service = ClinicalMemoryService()
