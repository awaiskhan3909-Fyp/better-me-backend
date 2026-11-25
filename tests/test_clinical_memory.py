import sys
import os
import uuid
import unittest
from fastapi.testclient import TestClient

# Python 3.11 typing patch before SQLAlchemy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.db.database
from app.db.database import SessionLocal
from app.db.models import User, Conversation, EpisodicTherapyMemory, PatientClinicalProfile
from app.services.clinical_memory_service import clinical_memory_service
from app.main import app


class TestClinicalMemory(unittest.TestCase):
    """
    Task 2.4: Verification of Longitudinal Clinical Memory.
    Ensures that presenting triggers, cognitive distortions, and episodic memories
    from completed sessions are persisted and successfully injected into future sessions.
    """

    @classmethod
    def setUpClass(cls):
        cls.db = SessionLocal()
        cls.client = TestClient(app)

        cls.user_email = f"clinical_test_{uuid.uuid4().hex[:8]}@betterme.app"
        cls.test_user = User(
            id=uuid.uuid4(),
            email=cls.user_email,
            password_hash="test_secure_hash",
            full_name="Alex Miller"
        )
        cls.db.add(cls.test_user)
        cls.db.commit()
        cls.db.refresh(cls.test_user)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_clinical_memory_continuity_across_sessions(self):
        """
        Creates a session mentioning 'my exam and my brother', ends it,
        starts a new session for the same user, and asserts that the memory briefing
        contains that context ('exam' and 'brother').
        """
        # Step 1: Create Session 1 for user
        res1 = self.client.post("/api/conversations", json={"user_id": str(self.test_user.id)})
        self.assertIn(res1.status_code, [200, 201])
        conv1_id = res1.json()["id"]

        # User shares situation mentioning "my exam and my brother"
        user_message = "I am really stressed about my exam and my brother is not helping at all."
        msg_res = self.client.post(
            f"/api/conversations/{conv1_id}/messages",
            json={"text": user_message}
        )
        self.assertIn(msg_res.status_code, [200, 201])

        # Step 2: Formally complete/end Session 1
        complete_res = self.client.post(f"/api/conversations/{conv1_id}/complete")
        self.assertIn(complete_res.status_code, [200, 201])
        report_data = complete_res.json()
        self.assertEqual(report_data.get("status"), "completed")

        # Verify DB episodic memory and clinical profile were persisted
        episode = self.db.query(EpisodicTherapyMemory).filter(
            EpisodicTherapyMemory.user_id == self.test_user.id
        ).first()
        self.assertIsNotNone(episode, "Episodic therapy memory must be saved to DB")
        self.assertIn("exam", episode.situation_context.lower())
        self.assertIn("brother", episode.situation_context.lower())

        profile = self.db.query(PatientClinicalProfile).filter(
            PatientClinicalProfile.user_id == self.test_user.id
        ).first()
        self.assertIsNotNone(profile, "Patient clinical profile must be created")
        self.assertIn("Academic & Exams", profile.primary_triggers)
        self.assertIn("Social & Relationships", profile.primary_triggers)

        # Step 3: Start a new session (Session 2) for the exact same user
        res2 = self.client.post("/api/conversations", json={"user_id": str(self.test_user.id)})
        self.assertIn(res2.status_code, [200, 201])
        conv2_id = res2.json()["id"]
        self.assertNotEqual(conv1_id, conv2_id)

        # Step 4: Verify memory briefing generated for the new session
        briefing = clinical_memory_service.build_clinical_memory_briefing(
            user_id=self.test_user.id,
            current_text="I am trying to prepare for my upcoming exam again.",
            db=self.db
        )

        # Assert briefing contains the required longitudinal context
        self.assertIn("--- LONGITUDINAL PATIENT CLINICAL MEMORY ---", briefing)
        self.assertIn("exam", briefing.lower(), "Briefing must contain 'exam' trigger context")
        self.assertIn("brother", briefing.lower(), "Briefing must contain 'brother' episodic context")
        self.assertIn("Academic & Exams", briefing)
        self.assertIn("Social & Relationships", briefing)


if __name__ == "__main__":
    unittest.main()
