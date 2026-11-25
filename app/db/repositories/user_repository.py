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

    def create_user(
        self,
        email: str,
        password_hash: str,
        full_name: Optional[str] = None,
        role: str = "user",
        user_id: Optional[uuid.UUID] = None,
        commit: bool = True
    ) -> User:
        user_kwargs = dict(
            email=email.strip().lower(),
            password_hash=password_hash,
            full_name=full_name.strip() if full_name else None,
            role=role
        )
        if user_id is not None:
            user_kwargs["id"] = user_id
        user = User(**user_kwargs)
        self.db.add(user)
        if commit:
            self.db.commit()
            self.db.refresh(user)
        else:
            self.db.flush()
        return user

    def upsert_supabase_user(
        self,
        email: str,
        user_id: Optional[uuid.UUID] = None,
        full_name: Optional[str] = None,
        password_hash: Optional[str] = None,
        commit: bool = True
    ) -> User:
        clean_email = email.strip().lower()
        user = None
        if user_id is not None:
            user = self.get_by_id(user_id)
        if not user:
            user = self.get_by_email(clean_email)

        if user:
            changed = False
            if password_hash and user.password_hash != password_hash:
                user.password_hash = password_hash
                changed = True
            if full_name and full_name.strip() and not user.full_name:
                user.full_name = full_name.strip()
                changed = True
            if changed and commit:
                self.db.commit()
                self.db.refresh(user)
            return user

        return self.create_user(
            email=clean_email,
            password_hash=password_hash or "supabase_managed",
            full_name=full_name,
            user_id=user_id,
            commit=commit
        )

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
