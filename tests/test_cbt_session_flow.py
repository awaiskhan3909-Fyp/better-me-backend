import sys
import unittest
import uuid
from pathlib import Path

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import app.db.database
from app.db.database import Base, engine, SessionLocal
from app.db.models import User, Conversation, Message, ConversationSessionState
from app.services.session_state_service import (
    session_state_service,
    TherapyStage,
    UserIntent
)
from app.services.response_decision_service import (
    response_decision_service,
    DecisionContext,
    ResponseStrategy,
    DecisionPriority
)
from app.services.distortion_service import distortion_service, CANONICAL_DISTORTIONS
from app.services.llm.response_parser import response_parser
from app.services.llm.prompt_builder import prompt_builder
from app.services.llm.fallback_service import fallback_service
from app.schemas.response import SafetyPrediction, DistortionPrediction, EntityItem


class TestCBTSessionFlow(unittest.TestCase):
    """
    Comprehensive test suite for Better Me CBT Conversation Intelligence.
    Tests clinical pacing, question deduplication, raw label sanitization,
    intent classification, off-topic handling, and structured closure.
    """

    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)

    def setUp(self):
        self.db = SessionLocal()
        self.test_user = self.db.query(User).filter(User.email == "test_cbt_user@example.com").first()
        if not self.test_user:
            self.test_user = User(
                email="test_cbt_user@example.com",
                password_hash="test_hash",
                full_name="CBT Test Subject"
            )
            self.db.add(self.test_user)
            self.db.commit()
            self.db.refresh(self.test_user)

        self.conversation = Conversation(
            user_id=self.test_user.id,
            title="Automated CBT Test Session"
        )
        self.db.add(self.conversation)
        self.db.commit()
        self.db.refresh(self.conversation)

    def tearDown(self):
        if self.conversation:
            self.db.query(ConversationSessionState).filter(
                ConversationSessionState.conversation_id == self.conversation.id
            ).delete()
            self.db.query(Message).filter(
                Message.conversation_id == self.conversation.id
            ).delete()
            self.db.query(Conversation).filter(
                Conversation.id == self.conversation.id
            ).delete()
            self.db.commit()
        self.db.close()

    def test_raw_label_sanitization(self):
        """Problem 7: Raw labels like LABEL_3 and LABEL_5 must NEVER appear."""
        self.assertEqual(response_decision_service.sanitize_distortion_label("LABEL_1"), "Catastrophizing")
        self.assertEqual(response_decision_service.sanitize_distortion_label("LABEL_3"), "Fortune Telling")
        self.assertEqual(response_decision_service.sanitize_distortion_label("LABEL_5"), "Overgeneralization")
        self.assertEqual(response_decision_service.sanitize_distortion_label("LABEL_99"), "None")
        self.assertEqual(response_decision_service.sanitize_distortion_label("None"), "None")

        # Test distortion_service canonical mapping
        self.assertNotIn("LABEL_0", distortion_service.label_map.values())
        self.assertIn("Catastrophizing", distortion_service.label_map.values())

    def test_single_question_rule_enforcement(self):
        """Single Question Rule: Output must strictly contain at most ONE question mark."""
        multi_q_text = (
            "I understand how disappointed you are. "
            "How long have you felt this way? "
            "What do you think was the root cause? "
            "Are you ready to explore an alternative view?"
        )
        is_valid, sanitized, err = response_parser.validate_and_sanitize(multi_q_text)
        self.assertTrue(is_valid)
        self.assertEqual(sanitized.count("?"), 1, "Output must contain at most one question mark.")
        self.assertTrue(sanitized.endswith("Are you ready to explore an alternative view?"))

    def test_leakage_and_stage_direction_stripping(self):
        """Eliminate prompt leakage and stage annotations like (Socratic questioning...)."""
        dirty_text = (
            "Directive: CBT Support\n"
            "### Response:\n"
            "(Validation and empathetic response)\n"
            "I hear how heavy this feels right now.\n"
            "(Socratic questioning to identify distortion)\n"
            "What evidence supports the thought that you will fail again?"
        )
        is_valid, sanitized, err = response_parser.validate_and_sanitize(dirty_text)
        self.assertTrue(is_valid)
        self.assertNotIn("Directive:", sanitized)
        self.assertNotIn("###", sanitized)
        self.assertNotIn("(Validation", sanitized)
        self.assertNotIn("(Socratic questioning", sanitized)
        self.assertIn("What evidence supports the thought that you will fail again?", sanitized)

    def test_early_thanks_does_not_close_session(self):
        """Problem 3: Saying 'thanks' early in the session must NOT prematurely close the session."""
        state = session_state_service.get_or_create_session_state(self.conversation.id, self.db)
        state.current_stage = TherapyStage.PROBLEM_EXPLORATION.value
        state.total_turns = 2

        safe_pred = SafetyPrediction(risk_level="Safe", needs_safety_alert=False, probabilities={"Safe": 0.99})
        dist_pred = DistortionPrediction(predicted_class="None", confidence=0.99, all_probabilities={"None": 0.99})

        # User says: "Thanks for asking" or "thank you"
        dec = response_decision_service.evaluate_decision(DecisionContext(
            user_text="Thank you for listening to me",
            safety_result=safe_pred,
            distortion_result=dist_pred,
            entities=[],
            session_state=state
        ))
        self.assertNotEqual(dec.therapy_stage, TherapyStage.CLOSURE.value, "Early thanks must NOT close the session!")
        self.assertEqual(dec.reason, "polite_acknowledgment_continue")

    def test_explicit_exit_request_triggers_closure(self):
        """Explicit exit requests ('I have to go now', 'bye') must trigger structured closure."""
        state = session_state_service.get_or_create_session_state(self.conversation.id, self.db)
        state.total_turns = 3

        safe_pred = SafetyPrediction(risk_level="Safe", needs_safety_alert=False, probabilities={"Safe": 0.99})
        dist_pred = DistortionPrediction(predicted_class="None", confidence=0.99, all_probabilities={"None": 0.99})

        dec = response_decision_service.evaluate_decision(DecisionContext(
            user_text="I have to go now, thank you",
            safety_result=safe_pred,
            distortion_result=dist_pred,
            entities=[],
            session_state=state
        ))
        self.assertEqual(dec.therapy_stage, TherapyStage.CLOSURE.value)
        self.assertEqual(dec.reason, "session_summary_and_closure")

    def test_off_topic_reply_handling(self):
        """Problem 4: Off-topic or casual interjections (pizza, weather) must be bridged back without closing session."""
        state = session_state_service.get_or_create_session_state(self.conversation.id, self.db)
        state.current_stage = TherapyStage.PROBLEM_EXPLORATION.value
        state.total_turns = 2

        safe_pred = SafetyPrediction(risk_level="Safe", needs_safety_alert=False, probabilities={"Safe": 0.99})
        dist_pred = DistortionPrediction(predicted_class="None", confidence=0.99, all_probabilities={"None": 0.99})

        dec = response_decision_service.evaluate_decision(DecisionContext(
            user_text="I am eating a pizza right now",
            safety_result=safe_pred,
            distortion_result=dist_pred,
            entities=[],
            session_state=state
        ))
        self.assertEqual(dec.reason, "off_topic_bridge")
        self.assertEqual(dec.therapy_stage, TherapyStage.PROBLEM_EXPLORATION.value)
        self.assertNotEqual(dec.therapy_stage, TherapyStage.CLOSURE.value)

    def test_mid_session_greeting_does_not_restart_intro(self):
        """Problem 5: Saying 'hey' or 'hello' in mid-session must NOT restart the introduction from scratch."""
        state = session_state_service.get_or_create_session_state(self.conversation.id, self.db)
        state.current_stage = TherapyStage.PROBLEM_EXPLORATION.value
        state.total_turns = 3

        safe_pred = SafetyPrediction(risk_level="Safe", needs_safety_alert=False, probabilities={"Safe": 0.99})
        dist_pred = DistortionPrediction(predicted_class="None", confidence=0.99, all_probabilities={"None": 0.99})

        dec = response_decision_service.evaluate_decision(DecisionContext(
            user_text="Hey, are you there?",
            safety_result=safe_pred,
            distortion_result=dist_pred,
            entities=[],
            session_state=state
        ))
        self.assertNotEqual(dec.therapy_stage, TherapyStage.CHECK_IN.value)
        self.assertEqual(dec.reason, "mid_session_greeting_acknowledgment")

        # Fallback text must NOT say "I am Better Me"
        text, _ = fallback_service.generate_fallback(
            user_text="Hey",
            strategy=dec.strategy.value,
            decision_reason=dec.reason
        )
        self.assertNotIn("I am Better Me, your supportive", text)

    def test_frustration_and_negative_feedback(self):
        """Problem 4: Frustrated user ('Stop asking so many questions') must be handled with humility."""
        state = session_state_service.get_or_create_session_state(self.conversation.id, self.db)
        safe_pred = SafetyPrediction(risk_level="Safe", needs_safety_alert=False, probabilities={"Safe": 0.99})
        dist_pred = DistortionPrediction(predicted_class="None", confidence=0.99, all_probabilities={"None": 0.99})

        dec = response_decision_service.evaluate_decision(DecisionContext(
            user_text="Stop asking so many questions! It's annoying.",
            safety_result=safe_pred,
            distortion_result=dist_pred,
            entities=[],
            session_state=state
        ))
        self.assertEqual(dec.reason, "user_negative_feedback")
        self.assertEqual(dec.therapy_stage, "feedback_alignment")

        text, _ = fallback_service.generate_fallback(
            user_text="Stop asking questions",
            strategy=dec.strategy.value,
            decision_reason=dec.reason
        )
        self.assertIn("hear your frustration", text.lower())

    def test_confusion_handling(self):
        """Problem 4: User confusion must trigger clear plain language explanation."""
        intent = session_state_service.classify_user_intent("What do you mean by that? I don't get it.")
        self.assertEqual(intent, UserIntent.CONFUSION)

        safe_pred = SafetyPrediction(risk_level="Safe", needs_safety_alert=False, probabilities={"Safe": 0.99})
        dist_pred = DistortionPrediction(predicted_class="None", confidence=0.99, all_probabilities={"None": 0.99})

        decision = response_decision_service.evaluate_decision(DecisionContext(
            user_text="What do you mean by that?",
            safety_result=safe_pred,
            distortion_result=dist_pred,
            entities=[]
        ))
        self.assertEqual(decision.reason, "user_confusion_clarification")

    def test_structured_closure_summary_and_homework(self):
        """Closure must include structured summary, homework experiment, and next session goal."""
        text, _ = fallback_service.generate_fallback(
            user_text="Thank you, I have to go now",
            strategy="normal_conversation",
            decision_reason="session_summary_and_closure",
            therapy_stage="closure"
        )
        self.assertIn("Session Summary", text)
        self.assertIn("Homework", text)
        self.assertIn("Next Session Goal", text)

    def test_controlled_multi_stage_progression_pacing(self):
        """Problem 1: Verifies multi-stage pacing requires exploration before jumping to restructuring."""
        state = session_state_service.get_or_create_session_state(self.conversation.id, self.db)

        safe_pred = SafetyPrediction(risk_level="Safe", needs_safety_alert=False, probabilities={"Safe": 0.99})
        dist_pred = DistortionPrediction(predicted_class="Catastrophizing", confidence=0.85, all_probabilities={"Catastrophizing": 0.85})

        # Turn 1: Greeting
        dec1 = response_decision_service.evaluate_decision(DecisionContext(
            user_text="Hello",
            safety_result=safe_pred,
            distortion_result=dist_pred,
            entities=[],
            session_state=state
        ))
        self.assertEqual(dec1.therapy_stage, TherapyStage.CHECK_IN.value)
        session_state_service.record_turn(state, TherapyStage(dec1.therapy_stage), "Hello", "How are you feeling today?")

        # Turn 2: User states initial problem -> EXPLORATION
        dec2 = response_decision_service.evaluate_decision(DecisionContext(
            user_text="I failed my programming exam and I am stressed.",
            safety_result=safe_pred,
            distortion_result=dist_pred,
            entities=[],
            session_state=state
        ))
        self.assertEqual(dec2.therapy_stage, TherapyStage.PROBLEM_EXPLORATION.value)
        session_state_service.record_turn(state, TherapyStage(dec2.therapy_stage), "I failed...", "What specifically happened?")

        # Turn 3: User answers context -> Still in EXPLORATION (controlled pacing to gather who/where/when)
        dec3 = response_decision_service.evaluate_decision(DecisionContext(
            user_text="It was yesterday in the afternoon class.",
            safety_result=safe_pred,
            distortion_result=dist_pred,
            entities=[],
            session_state=state
        ))
        self.assertEqual(dec3.therapy_stage, TherapyStage.PROBLEM_EXPLORATION.value, "Controlled pacing requires context exploration before jumping!")
        session_state_service.record_turn(state, TherapyStage(dec3.therapy_stage), "It was yesterday...", "Who was involved?")

        # Turn 4: User shares somatic symptom / distress -> Advances to EMOTION_AND_BODY
        dec4 = response_decision_service.evaluate_decision(DecisionContext(
            user_text="My heart was racing and I felt a tight knot in my chest.",
            safety_result=safe_pred,
            distortion_result=dist_pred,
            entities=[],
            session_state=state
        ))
        self.assertEqual(dec4.therapy_stage, TherapyStage.EMOTION_AND_BODY.value)
        session_state_service.record_turn(state, TherapyStage(dec4.therapy_stage), "My heart was racing...", "When your heart raced, what thought ran through your mind?")

        # Turn 5: User shares automatic thought -> Advances to THOUGHT_IDENTIFICATION
        dec5 = response_decision_service.evaluate_decision(DecisionContext(
            user_text="I thought to myself: I am a total failure and my entire future is ruined.",
            safety_result=safe_pred,
            distortion_result=dist_pred,
            entities=[],
            session_state=state
        ))
        self.assertEqual(dec5.therapy_stage, TherapyStage.THOUGHT_IDENTIFICATION.value)
        session_state_service.record_turn(state, TherapyStage(dec5.therapy_stage), "I thought I am a total failure...", "What factual evidence supports that thought?")

        # Turn 6: Advances to DISTORTION_ANALYSIS (gated collaborative examination)
        dec6 = response_decision_service.evaluate_decision(DecisionContext(
            user_text="I felt like everyone else understood it except me.",
            safety_result=safe_pred,
            distortion_result=dist_pred,
            entities=[],
            session_state=state
        ))
        self.assertEqual(dec6.therapy_stage, TherapyStage.DISTORTION_ANALYSIS.value)
        session_state_service.record_turn(state, TherapyStage(dec6.therapy_stage), "I felt like everyone...", "What facts contradict that?")

        # Turn 7: Advances to SOCRATIC_INTERVENTION
        dec7 = response_decision_service.evaluate_decision(DecisionContext(
            user_text="I know I passed the previous assignments, but this test made me panic.",
            safety_result=safe_pred,
            distortion_result=dist_pred,
            entities=[],
            session_state=state
        ))
        self.assertEqual(dec7.therapy_stage, TherapyStage.SOCRATIC_INTERVENTION.value)
        session_state_service.record_turn(state, TherapyStage(dec7.therapy_stage), "I know I passed...", "What is a more balanced thought?")

        # Turn 8: Advances to ACTION_PLAN
        dec8 = response_decision_service.evaluate_decision(DecisionContext(
            user_text="What should I do right now to prepare better?",
            safety_result=safe_pred,
            distortion_result=dist_pred,
            entities=[],
            session_state=state
        ))
        self.assertEqual(dec8.therapy_stage, TherapyStage.ACTION_PLAN.value)

    def test_question_deduplication(self):
        """Problem 2: Deduplication prevents picking previously asked questions."""
        state = session_state_service.get_or_create_session_state(self.conversation.id, self.db)
        q1 = "When this situation happened, what emotions felt the strongest—like anxiety, sadness, anger, or feeling overwhelmed?"
        state.asked_questions = [q1]

        self.assertTrue(session_state_service.is_question_duplicate(state, q1))
        self.assertFalse(session_state_service.is_question_duplicate(state, "What small step can you take today?"))

    def test_safety_triage_override(self):
        """Safety override must trigger with highest priority regardless of current stage."""
        state = session_state_service.get_or_create_session_state(self.conversation.id, self.db)
        state.current_stage = TherapyStage.DISTORTION_ANALYSIS.value

        safe_pred = SafetyPrediction(risk_level="High Risk", needs_safety_alert=True, probabilities={"High Risk": 0.99})
        dist_pred = DistortionPrediction(predicted_class="Catastrophizing", confidence=0.90, all_probabilities={"Catastrophizing": 0.90})

        decision = response_decision_service.evaluate_decision(DecisionContext(
            user_text="I can't go on anymore, I want to end my life",
            safety_result=safe_pred,
            distortion_result=dist_pred,
            entities=[],
            session_state=state
        ))

        self.assertEqual(decision.strategy, ResponseStrategy.SAFETY_RESPONSE)
        self.assertEqual(decision.priority, DecisionPriority.CRITICAL)
        self.assertEqual(decision.safety_risk_level, "High Risk")


if __name__ == "__main__":
    unittest.main()
