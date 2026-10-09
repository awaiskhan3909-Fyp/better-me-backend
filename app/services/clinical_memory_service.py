import re
import uuid
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from app.db.models import (
    User,
    PatientClinicalProfile,
    EpisodicTherapyMemory,
    UserIntakeAssessment
)


class ClinicalMemoryService:
    """
    Clinical Longitudinal Memory Service.
    Maintains persistent patient profiles, recurring distortion habits,
    past breakthrough episodic memories, and behavioral homework tracking.
    """

    TRIGGER_KEYWORDS = {
        "Academic & Exams": ["exam", "test", "quiz", "grade", "gpa", "study", "studying", "marks", "fail", "failed", "assignment", "deadline", "university", "college"],
        "Career & Workplace": ["job", "interview", "boss", "work", "office", "career", "promotion", "resume", "colleague"],
        "Social & Relationships": ["friend", "friends", "partner", "relationship", "breakup", "lonely", "rejected", "rejection", "ignoring", "family", "parents"],
        "Health & Somatic": ["sleep", "insomnia", "tired", "exhausted", "headache", "panic", "heartbeat", "health", "sick"],
        "Self-Worth & Performance": ["failure", "useless", "worthless", "mistake", "not good enough", "loser", "perfectionist"]
    }

    def get_or_create_profile(self, user_id: uuid.UUID, db: Session) -> PatientClinicalProfile:
        profile = db.query(PatientClinicalProfile).filter(PatientClinicalProfile.user_id == user_id).first()
        if not profile:
            # Check if intake assessment exists to pre-populate baseline triggers
            intake = db.query(UserIntakeAssessment).filter(UserIntakeAssessment.user_id == user_id).first()
            initial_triggers = intake.primary_focus if intake else []
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

        # 2. Extract and Update Triggers
        text_lower = user_text.lower()
        triggers = list(profile.primary_triggers or [])
        for category, kws in self.TRIGGER_KEYWORDS.items():
            if any(re.search(rf"\b{kw}\b", text_lower) for kw in kws):
                if category not in triggers:
                    triggers.append(category)
        profile.primary_triggers = triggers

        # 3. Log Episodic Therapy Memory if significant distortion reframed
        if detected_distortion and detected_distortion not in ["None", "Normal", "no distortion"]:
            # Check if similar episode already logged in this conversation to avoid duplicates
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
                    situation_context=user_text[:180],
                    distorted_thought=user_text[:200],
                    distortion_type=detected_distortion,
                    rational_reframe=reframe_text[:250],
                    breakthrough_notes=f"Reframed {detected_distortion} with Socratic questioning."
                )
                db.add(episode)

        profile.updated_at = datetime.now(timezone.utc)
        db.commit()

    def build_clinical_memory_briefing(
        self,
        user_id: Optional[uuid.UUID],
        current_text: str,
        db: Session
    ) -> str:
        """
        Builds a compact, actionable clinical prompt block for the LLM.
        """
        if not user_id:
            return ""

        profile = db.query(PatientClinicalProfile).filter(PatientClinicalProfile.user_id == user_id).first()
        intake = db.query(UserIntakeAssessment).filter(UserIntakeAssessment.user_id == user_id).first()

        lines = ["--- LONGITUDINAL PATIENT CLINICAL MEMORY ---"]

        # Primary Goal & Baseline
        if intake and intake.primary_goal:
            lines.append(f"Patient Primary Goal: {intake.primary_goal}")

        # Known Triggers
        if profile and profile.primary_triggers:
            lines.append(f"Recurring Triggers: {', '.join(profile.primary_triggers)}")

        # Dominant Distortions History
        if profile and profile.dominant_distortions:
            sorted_dist = sorted(profile.dominant_distortions.items(), key=lambda x: x[1], reverse=True)[:3]
            dist_str = ", ".join([f"{k} ({v}x)" for k, v in sorted_dist])
            lines.append(f"Dominant Historical Distortions: {dist_str}")

        # Active Homework
        if profile and profile.active_homework:
            lines.append(f"Active Behavioral Homework: {profile.active_homework}")

        # Search for Relevant Past Breakthroughs / Episodic Memories
        episodes = (
            db.query(EpisodicTherapyMemory)
            .filter(EpisodicTherapyMemory.user_id == user_id)
            .order_by(EpisodicTherapyMemory.created_at.desc())
            .limit(5)
            .all()
        )

        relevant_episode = None
        current_lower = current_text.lower()

        # Find episode matching current keywords
        for ep in episodes:
            ep_words = re.findall(r"\w+", ep.situation_context.lower())
            overlap = [w for w in ep_words if len(w) > 4 and w in current_lower]
            if overlap:
                relevant_episode = ep
                break

        if not relevant_episode and episodes:
            relevant_episode = episodes[0]

        if relevant_episode:
            lines.append("Relevant Past Episode:")
            lines.append(f"  • Past situation: \"{relevant_episode.situation_context[:100]}\"")
            lines.append(f"  • Reframed thought: \"{relevant_episode.rational_reframe[:120]}\"")
            lines.append("CLINICAL DIRECTIVE: If natural, reference their past resilience or learned coping strategy to foster self-efficacy.")

        lines.append("--------------------------------------------")
        return "\n".join(lines)


clinical_memory_service = ClinicalMemoryService()
