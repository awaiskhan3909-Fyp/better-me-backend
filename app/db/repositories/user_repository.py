import uuid
from typing import Optional
from sqlalchemy.orm import Session
from app.db.models import User


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_user(self, email: str, password_hash: str, full_name: Optional[str] = None, commit: bool = True) -> User:
        user = User(
            email=email,
            password_hash=password_hash,
            full_name=full_name
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
        return self.db.query(User).filter(User.email == email).first()

    def get_or_create_default_user(self, commit: bool = True) -> User:
        default_email = "demo_user@betterme.app"
        user = self.get_by_email(default_email)
        if not user:
            user = self.create_user(
                email=default_email,
                password_hash="demo_hashed_password_123",
                full_name="Better Me Default User",
                commit=commit
            )
        return user
