import sys
import os
import uuid
import unittest
from datetime import datetime

# Python 3.11 typing patch before SQLAlchemy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.db.database
from app.db.database import SessionLocal
from app.db.models import Conversation, ConversationSessionState, User, Message, MessageAnalysis
from app.services.session_state_service import session_state_service, TherapyStage, UserIntent
from app.services.response_decision_service import (
    response_decision_service, DecisionContext, ResponseStrategy
)
from app.schemas.response import SafetyPrediction, DistortionPrediction, EntityItem
from app.services.pdf_report_service import pdf_report_service
from fastapi.testclient import TestClient
from app.main import app


class TestCBTFinalizationAndPDFReport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = SessionLocal()
        cls.client = TestClient(app)

        # Get or create test user
        cls.test_user = cls.db.query(User).filter(User.email == "test_patient_pdf@betterme.app").first()
        if not cls.test_user:
            cls.test_user = User(
                id=uuid.uuid4(),
                email="test_patient_pdf@betterme.app",
                password_hash="test_hash_secret",
                full_name="Sarah Jenkins"
            )
            cls.db.add(cls.test_user)
            cls.db.commit()
            cls.db.refresh(cls.test_user)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_01_perfectionism_intent_detection(self):
        """Tests that perfectionistic statements trigger PERFECTIONISM intent and good-enough reframing."""
        intent = session_state_service.classify_user_intent("Everything has to be 100% perfect or I am an utter failure")
        self.assertEqual(intent, UserIntent.PERFECTIONISM)

        # Roman Urdu perfectionism
        urdu_intent = session_state_service.classify_user_intent("sab kuch perfect hona chahiye ghalti ki gunjaish nahi")
        self.assertEqual(urdu_intent, UserIntent.PERFECTIONISM)

        # Response Decision
        context = DecisionContext(
            user_text="Everything has to be 100% perfect or I am an utter failure",
            safety_result=SafetyPrediction(risk_level="Safe", needs_safety_alert=False, probabilities={"Safe": 0.99}),
            distortion_result=DistortionPrediction(predicted_class="All-or-Nothing Thinking", confidence=0.88, all_probabilities={}),
            entities=[]
        )
        decision = response_decision_service.evaluate_decision(context)
        self.assertEqual(decision.strategy, ResponseStrategy.CBT_SUPPORT)
        self.assertEqual(decision.reason, "perfectionism_exploration")
        self.assertEqual(decision.distortion_detected, "All-or-Nothing Thinking")

    def test_02_uncertainty_intent_detection(self):
        """Tests that uncertainty/ambivalence triggers gentle grounding without rushing session stages."""
        intent = session_state_service.classify_user_intent("I'm not sure, I really don't know what to do")
        self.assertEqual(intent, UserIntent.UNCERTAINTY)

        urdu_intent = session_state_service.classify_user_intent("mujhe kuch samajh nahi aa raha shayad main galat hoon")
        self.assertEqual(urdu_intent, UserIntent.UNCERTAINTY)

        context = DecisionContext(
            user_text="I'm not sure, I really don't know what to do",
            safety_result=SafetyPrediction(risk_level="Safe", needs_safety_alert=False, probabilities={"Safe": 0.99}),
            distortion_result=DistortionPrediction(predicted_class="None", confidence=0.0, all_probabilities={}),
            entities=[]
        )
        decision = response_decision_service.evaluate_decision(context)
        self.assertEqual(decision.strategy, ResponseStrategy.EXPLORATORY_FOLLOW_UP)
        self.assertEqual(decision.reason, "uncertainty_grounding")

    def test_03_prevent_premature_closure(self):
        """Ensures that brief statements, uncertainty, or mild thanks do not abruptly terminate early sessions."""
        state = ConversationSessionState(
            conversation_id=uuid.uuid4(),
            current_stage=TherapyStage.PROBLEM_EXPLORATION.value,
            stage_turn_count=1,
            total_turns=2,
            asked_questions=[],
            session_goals=[],
            distortion_evidence=[],
            metadata_json={"explored_dimensions": {}, "identified_emotions": []}
        )
        next_stage = session_state_service.progress_session_stage(
            state=state,
            user_text="thanks",
            user_intent=UserIntent.POLITE_ACKNOWLEDGMENT,
            distortion_pred=DistortionPrediction(predicted_class="None", confidence=0.0, all_probabilities={}),
            is_evidence_gated=False
        )
        self.assertEqual(next_stage, TherapyStage.PROBLEM_EXPLORATION)

    def test_04_dynamic_homework_customization(self):
        """Verifies that homework dynamically tailors to the patient's problem context."""
        state = ConversationSessionState(
            conversation_id=uuid.uuid4(),
            current_stage=TherapyStage.ACTION_PLAN.value,
            stage_turn_count=1,
            total_turns=6,
            identified_trigger="Final university exam and course grade worries",
            asked_questions=[],
            session_goals=[],
            distortion_evidence=[],
            metadata_json={"explored_dimensions": {}, "identified_emotions": []}
        )
        session_state_service.record_turn(
            state=state,
            next_stage=TherapyStage.CLOSURE,
            user_text="I will try to take a breath and study",
            ai_question_asked=None
        )
        meta = state.metadata_json
        self.assertIn("homework", meta)
        self.assertIn("Study", meta["homework"])
        self.assertIn("exam", meta["next_session_goal"].lower())

    def test_05_pdf_report_generation_complete_session(self):
        """Tests end-to-end PDF report generation for a complete CBT therapy session."""
        conv = Conversation(
            id=uuid.uuid4(),
            user_id=self.test_user.id,
            title="Comprehensive CBT Session Test",
            current_risk_level="Safe"
        )
        self.db.add(conv)
        self.db.commit()

        # Add user and AI messages
        m1 = Message(conversation_id=conv.id, sender_type="user", content="I am terrified about failing my upcoming board exam.", sequence_number=1)
        m2 = Message(conversation_id=conv.id, sender_type="ai", content="I hear your concern. What thoughts run through your mind?", sequence_number=2)
        m3 = Message(conversation_id=conv.id, sender_type="user", content="If I fail, my entire life is ruined forever.", sequence_number=3)
        m4 = Message(conversation_id=conv.id, sender_type="ai", content="Let's look at that catastrophic prediction together.", sequence_number=4)
        self.db.add_all([m1, m2, m3, m4])

        state = ConversationSessionState(
            conversation_id=conv.id,
            current_stage=TherapyStage.CLOSURE.value,
            stage_turn_count=2,
            total_turns=8,
            identified_trigger="Upcoming medical board licensure examination",
            automatic_thought="If I fail, my entire life is ruined forever.",
            identified_distortion="Catastrophizing",
            distortion_confidence=0.92,
            distortion_evidence=["If I fail, my entire life is ruined forever."],
            evidence_gated=True,
            balanced_perspective="Failing a single exam would be difficult and disappointing, but it is retakeable and does not permanently determine my worth as a person.",
            action_plan="Practice a 20-minute timed study block using the good-enough framework.",
            metadata_json={
                "explored_dimensions": {"situation": True, "emotions": True, "physiology": True, "thought": True},
                "identified_emotions": ["Performance Dread", "Anticipatory Anxiety", "Exhaustion"],
                "somatic_symptoms": ["Tightness in chest", "Muscle tension", "Restless sleep"],
                "homework": "Practice a 20-minute timed study block using the good-enough framework.",
                "next_session_goal": "Review study focus resilience and verify whether catastrophic assumptions held true."
            }
        )
        self.db.add(state)
        self.db.commit()
        self.db.refresh(conv)

        # Generate PDF Buffer
        pdf_buf = pdf_report_service.generate_pdf_buffer(conv, self.db)
        pdf_bytes = pdf_buf.getvalue()

        self.assertGreater(len(pdf_bytes), 10000, "PDF byte size should be substantial (>= 10 KB)")
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"), "Generated buffer must have valid PDF header signature")

    def test_06_pdf_report_missing_fields_graceful_fallback(self):
        """Verifies that an empty or minimally initialized session still produces a valid PDF without errors or hallucinations."""
        conv_empty = Conversation(
            id=uuid.uuid4(),
            user_id=self.test_user.id,
            title="Minimal Session",
            current_risk_level="Safe"
        )
        self.db.add(conv_empty)
        self.db.commit()
        self.db.refresh(conv_empty)

        # Generate PDF Buffer with zero messages and no session_state record
        pdf_buf = pdf_report_service.generate_pdf_buffer(conv_empty, self.db)
        pdf_bytes = pdf_buf.getvalue()

        self.assertGreater(len(pdf_bytes), 5000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

    def test_07_api_endpoint_summary(self):
        """Verifies GET /api/conversations/{id}/report/summary returns structured JSON data."""
        conv = self.db.query(Conversation).filter(Conversation.user_id == self.test_user.id).first()
        res = self.client.get(f"/api/conversations/{conv.id}/report/summary")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["conversation_id"], str(conv.id))
        self.assertIn("report_id", data)
        self.assertIn("patient_name", data)
        self.assertIn("balanced_thought", data)
        self.assertIn("homework", data)

    def test_08_api_endpoint_pdf_download(self):
        """Verifies GET /api/conversations/{id}/report streams application/pdf with attachment header."""
        conv = self.db.query(Conversation).filter(Conversation.user_id == self.test_user.id).first()
        res = self.client.get(f"/api/conversations/{conv.id}/report")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("content-type"), "application/pdf")
        self.assertIn("attachment", res.headers.get("content-disposition", ""))
        self.assertTrue(res.content.startswith(b"%PDF-"))

    def test_09_api_endpoint_unauthorized_user_isolation(self):
        """Ensures that passing a different user_id prevents downloading another user's private clinical report."""
        conv = self.db.query(Conversation).filter(Conversation.user_id == self.test_user.id).first()
        alien_user_id = str(uuid.uuid4())
        res = self.client.get(f"/api/conversations/{conv.id}/report?user_id={alien_user_id}")
        self.assertEqual(res.status_code, 403)
        self.assertIn("Unauthorized access", res.json()["detail"])


if __name__ == "__main__":
    unittest.main()

