import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories.user_repository import UserRepository
from app.schemas.auth import RegisterRequest, LoginRequest, AuthResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register_user(payload: RegisterRequest, db: Session = Depends(get_db)):
    user_repo = UserRepository(db)

    # Check if user already exists
    existing = user_repo.get_by_email(payload.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address already exists. Please log in."
        )

    hashed_pw = user_repo.hash_password(payload.password)
    new_user = user_repo.create_user(
        email=payload.email,
        password_hash=hashed_pw,
        full_name=payload.full_name
    )

    user_data = UserResponse(
        id=new_user.id,
        email=new_user.email,
        full_name=new_user.full_name,
        is_active=new_user.is_active,
        has_completed_intake=False,
        created_at=new_user.created_at
    )

    return AuthResponse(user=user_data, message="Account created successfully.")


@router.post("/login", response_model=AuthResponse)
def login_user(payload: LoginRequest, db: Session = Depends(get_db)):
    user_repo = UserRepository(db)

    user = user_repo.get_by_email(payload.email)
    if not user or not user_repo.verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    intake = user_repo.get_intake_assessment(user.id)
    has_completed_intake = bool(intake)

    user_data = UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        has_completed_intake=has_completed_intake,
        created_at=user.created_at
    )

    return AuthResponse(user=user_data, message="Logged in successfully.")


@router.get("/me/{user_id}", response_model=UserResponse)
def get_user_profile(user_id: uuid.UUID, db: Session = Depends(get_db)):
    user_repo = UserRepository(db)
    user = user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    intake = user_repo.get_intake_assessment(user.id)
    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        has_completed_intake=bool(intake),
        created_at=user.created_at
    )
