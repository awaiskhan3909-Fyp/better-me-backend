import uuid
from typing import Optional
from sqlalchemy.orm import Session
import hashlib
from app.db.models import User, UserIntakeAssessment
from app.schemas.auth import IntakeAssessmentRequest


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(f"bm_salt_{password}".encode("utf-8")).hexdigest()

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.hash_password(plain_password) == hashed_password

    def create_user(self, email: str, password_hash: str, full_name: Optional[str] = None, commit: bool = True) -> User:
        user = User(
            email=email.strip().lower(),
            password_hash=password_hash,
            full_name=full_name.strip() if full_name else None
        )
        self.db.add(user)
        if commit:
            self.db.commit()
            self.db.refresh(user)
        else:
            self.db.flush()
        return user

    def get_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        return self.db.query(User).filter(User.id == user_id).first()

    def get_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email.strip().lower()).first()

    def get_or_create_default_user(self, commit: bool = True) -> User:
        default_email = "demo_user@betterme.app"
        user = self.get_by_email(default_email)
        if not user:
            user = self.create_user(
                email=default_email,
                password_hash=self.hash_password("demo_password_123"),
                full_name="Better Me Default User",
                commit=commit
            )
        return user

    def save_intake_assessment(self, user_id: uuid.UUID, intake_data: IntakeAssessmentRequest) -> UserIntakeAssessment:
        existing = self.db.query(UserIntakeAssessment).filter(UserIntakeAssessment.user_id == user_id).first()
        if existing:
            existing.primary_focus = intake_data.primary_focus
            existing.distress_baseline = intake_data.distress_baseline
            existing.familiar_distortions = intake_data.familiar_distortions
            existing.primary_goal = intake_data.primary_goal
            existing.safety_acknowledged = intake_data.safety_acknowledged
            assessment = existing
        else:
            assessment = UserIntakeAssessment(
                user_id=user_id,
                primary_focus=intake_data.primary_focus,
                distress_baseline=intake_data.distress_baseline,
                familiar_distortions=intake_data.familiar_distortions,
                primary_goal=intake_data.primary_goal,
                safety_acknowledged=intake_data.safety_acknowledged,
            )
            self.db.add(assessment)

        self.db.commit()
        self.db.refresh(assessment)
        return assessment

    def get_intake_assessment(self, user_id: uuid.UUID) -> Optional[UserIntakeAssessment]:
        return self.db.query(UserIntakeAssessment).filter(UserIntakeAssessment.user_id == user_id).first()
