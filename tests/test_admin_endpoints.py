import sys
import unittest
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app
from app.db.database import Base, engine, SessionLocal
from app.db.models import (
    User,
    Conversation,
    Message,
    MessageAnalysis,
    AIResponse,
    UserIntakeAssessment,
    CBTThoughtRecord,
    ConversationSessionState,
    CBTExerciseSubmission,
    StressReliefActivity,
)
from app.db.repositories.user_repository import UserRepository


class TestAdminEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)
        cls.client = TestClient(app)
        cls.db = SessionLocal()

        # Clean up any previous test artifacts
        user_repo = UserRepository(cls.db)

        # 1. Seed Admin User
        admin_email = f"admin_{uuid.uuid4().hex[:6]}@betterme.app"
        cls.admin_user = user_repo.create_user(
            email=admin_email,
            password_hash=user_repo.hash_password("admin_pass_123"),
            full_name="Dr. Clinical Admin",
            role="admin"
        )

        # 2. Seed Regular Patient User
        regular_email = f"patient_{uuid.uuid4().hex[:6]}@betterme.app"
        cls.regular_user = user_repo.create_user(
            email=regular_email,
            password_hash=user_repo.hash_password("patient_pass_123"),
            full_name="Patient Sample",
            role="user"
        )

        now = datetime.now(timezone.utc)

        # 3. Seed Intake Assessment
        intake = UserIntakeAssessment(
            user_id=cls.regular_user.id,
            primary_focus=["Anxiety", "Workplace Stress"],
            distress_baseline=7,
            familiar_distortions=["Catastrophizing", "Mind Reading"],
            primary_goal="Stop worst-case thinking",
            safety_acknowledged=True,
            created_at=now
        )
        cls.db.add(intake)

        # 4. Seed Conversation
        conv = Conversation(
            user_id=cls.regular_user.id,
            title="Work Presentation Anxiety",
            current_risk_level="Medium",
            created_at=now,
            updated_at=now
        )
        cls.db.add(conv)
        cls.db.flush()

        # 5. Seed Messages
        msg_user = Message(
            conversation_id=conv.id,
            sender_type="user",
            content="I am terrified I will ruin the presentation.",
            sequence_number=1,
            created_at=now
        )
        msg_ai = Message(
            conversation_id=conv.id,
            sender_type="ai",
            content="It sounds like you are feeling intense pressure right now. What makes you think you will ruin it?",
            sequence_number=2,
            created_at=now
        )
        cls.db.add_all([msg_user, msg_ai])
        cls.db.flush()

        # 6. Seed Message Analysis
        analysis = MessageAnalysis(
            message_id=msg_user.id,
            conversation_id=conv.id,
            safety_risk_level="Medium",
            needs_safety_alert=False,
            safety_probabilities={"Safe": 0.3, "Low": 0.2, "Medium": 0.5},
            distortion_class="Catastrophizing",
            distortion_confidence=0.88,
            distortion_probabilities={"Catastrophizing": 0.88},
            entities=[],
            model_metadata={"engine": "bert"},
            created_at=now
        )
        cls.db.add(analysis)

        # 7. Seed AI Response Telemetry
        ai_resp = AIResponse(
            conversation_id=conv.id,
            user_message_id=msg_user.id,
            ai_message_id=msg_ai.id,
            response_source="conversational_llm",
            response_type="therapeutic_reframe",
            model_name="gemini-2.5-flash",
            response_content=msg_ai.content,
            created_at=now
        )
        cls.db.add(ai_resp)

        # 8. Seed Session State (CBT Funnel)
        state = ConversationSessionState(
            conversation_id=conv.id,
            current_stage="socratic_questioning",
            stage_turn_count=2,
            total_turns=2,
            identified_distortion="Catastrophizing",
            distortion_confidence=0.88,
            evidence_gated=True,
            created_at=now,
            updated_at=now
        )
        cls.db.add(state)

        # 9. Seed CBT Exercise Submission
        submission = CBTExerciseSubmission(
            user_id=cls.regular_user.id,
            distortion_category="Catastrophizing",
            exercise_id="catastrophizing-1",
            exercise_title="The Slide Deck Typo Panic",
            scenario="During an all-hands team presentation...",
            answers={"q1": "I thought I would be fired", "q2": "None"},
            is_completed=True,
            completed_at=now,
            created_at=now,
            updated_at=now
        )
        cls.db.add(submission)

        # 10. Seed CBT Thought Record
        thought_rec = CBTThoughtRecord(
            user_id=cls.regular_user.id,
            conversation_id=conv.id,
            situation="Upcoming client demo",
            automatic_thought="I am going to freeze completely.",
            initial_belief_rating=85,
            emotions={"anxiety": 8},
            distortion_type="Catastrophizing",
            evidence_for="I feel nervous",
            evidence_against="I practiced three times",
            balanced_thought="I may feel jittery at first, but I know the material.",
            outcome_belief_rating=25,
            outcome_emotions={"anxiety": 3},
            created_at=now,
            updated_at=now
        )
        cls.db.add(thought_rec)

        # 11. Seed Stress Relief Activity
        relief = StressReliefActivity(
            user_id=cls.regular_user.id,
            activity_name="4-7-8 Breathing",
            activity_type="breathing",
            duration_seconds=240,
            is_completed=True,
            completed_at=now,
            created_at=now
        )
        cls.db.add(relief)

        cls.db.commit()

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    # --- Test 1: Access Control (require_admin) ---
    def test_01_unauthenticated_request_rejected(self):
        """Unauthenticated requests without user identity or token should receive 401."""
        response = self.client.get("/api/admin/overview")
        self.assertEqual(response.status_code, 401)

    def test_02_regular_user_forbidden(self):
        """Users with role='user' must receive 403 Forbidden on all admin endpoints."""
        response = self.client.get(
            "/api/admin/overview",
            headers={"X-User-Id": str(self.regular_user.id)}
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("Administrator privileges required", response.json()["detail"])

    def test_03_admin_user_authorized(self):
        """Users with role='admin' should successfully access admin endpoints."""
        response = self.client.get(
            "/api/admin/overview",
            headers={"X-User-Id": str(self.admin_user.id)}
        )
        self.assertEqual(response.status_code, 200)

    # --- Test 2: Overview Endpoint ---
    def test_04_admin_overview_metrics(self):
        response = self.client.get(
            "/api/admin/overview?days=30",
            headers={"X-User-Id": str(self.admin_user.id)}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["total_registered_users"], 2)
        self.assertGreaterEqual(data["total_sessions"], 1)
        self.assertGreaterEqual(data["total_messages"], 2)
        self.assertGreaterEqual(data["total_practices_completed"], 3)
        self.assertIn("message_breakdown", data)
        self.assertIn("practice_breakdown", data)

    # --- Test 3: Users Cohort Privacy ---
    def test_05_admin_users_pseudonymization_and_privacy(self):
        """Verify that user records are pseudonymized and never leak emails, names, or content."""
        response = self.client.get(
            "/api/admin/users?days=30",
            headers={"X-User-Id": str(self.admin_user.id)}
        )
        self.assertEqual(response.status_code, 200)
        users = response.json()
        self.assertIsInstance(users, list)
        self.assertGreaterEqual(len(users), 1)

        for u in users:
            # Must have pseudonym_id starting with 'usr_'
            self.assertTrue(u["pseudonym_id"].startswith("usr_"))
            # MUST NOT leak sensitive fields
            self.assertNotIn("email", u)
            self.assertNotIn("full_name", u)
            self.assertNotIn("password_hash", u)
            self.assertNotIn("content", u)

    # --- Test 4: Safety Report ---
    def test_06_admin_safety_report(self):
        response = self.client.get(
            "/api/admin/safety?days=30",
            headers={"X-User-Id": str(self.admin_user.id)}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("risk_level_distribution", data)
        self.assertIn("Medium", data["risk_level_distribution"])
        self.assertIn("elevated_risk_conversations", data)

    # --- Test 5: Distortions Report ---
    def test_07_admin_distortions_report(self):
        response = self.client.get(
            "/api/admin/distortions?days=30",
            headers={"X-User-Id": str(self.admin_user.id)}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("detected_distortions_frequency", data)
        self.assertIn("Catastrophizing", data["detected_distortions_frequency"])
        self.assertGreater(data["average_confidence"], 0.0)
        self.assertIn("socratic_gating", data)
        self.assertIn("practices_by_distortion", data)

    # --- Test 6: Sessions Funnel ---
    def test_08_admin_sessions_funnel(self):
        response = self.client.get(
            "/api/admin/sessions-funnel?days=30",
            headers={"X-User-Id": str(self.admin_user.id)}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("cbt_funnel_stages", data)
        self.assertIn("session_completion_rate_percentage", data)

    # --- Test 7: AI Ops Telemetry ---
    def test_09_admin_ai_ops(self):
        response = self.client.get(
            "/api/admin/ai-ops?days=30",
            headers={"X-User-Id": str(self.admin_user.id)}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response_sources_breakdown", data)
        self.assertIn("active_models_breakdown", data)
        self.assertIn("offline_template_fallback_rate_percentage", data)

    # --- Test 8: Engagement Report ---
    def test_10_admin_engagement(self):
        response = self.client.get(
            "/api/admin/engagement?days=30",
            headers={"X-User-Id": str(self.admin_user.id)}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("intake", data)
        self.assertIn("thought_records_efficacy", data)
        # Verify belief rating reduction was computed
        self.assertGreater(data["thought_records_efficacy"]["average_belief_reduction"], 0)
        self.assertIn("stress_relief_adoption", data)
        self.assertIn("retention", data)

    # --- Test 9: System Status ---
    def test_11_admin_system_status(self):
        response = self.client.get(
            "/api/admin/system",
            headers={"X-User-Id": str(self.admin_user.id)}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["system_health"], "healthy")
        self.assertTrue(data["database_connected"])
        self.assertIn("table_row_counts", data)
        self.assertIn("ai_models_status", data)
        self.assertGreaterEqual(data["table_row_counts"]["users"], 2)


if __name__ == "__main__":
    unittest.main()
