import uuid
import os
from typing import Optional
from fastapi import Header, HTTPException, Depends, status, Query
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import User

ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "better_me_admin_secret_key_2026")


def require_admin(
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    authorization: Optional[str] = Header(default=None),
    x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key"),
    admin_user_id: Optional[str] = Query(default=None, alias="admin_user_id"),
    db: Session = Depends(get_db)
) -> User:
    """
    Validates that the caller has admin privileges.
    Checks:
    1. If X-Admin-Key matches ADMIN_SECRET_KEY, returns the admin user.
    2. If X-User-Id, admin_user_id, or Bearer token is provided, checks user.role == 'admin'.
    Raises 401 if unauthenticated, 403 if user is not an admin.
    """
    # 1. Master admin secret key check
    if x_admin_key and x_admin_key == ADMIN_SECRET_KEY:
        admin_user = db.query(User).filter(User.role == "admin").first()
        if admin_user:
            return admin_user
        return User(
            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
            email="admin@betterme.app",
            role="admin",
            full_name="System Admin"
        )

    target_id_str = x_user_id or admin_user_id

    # 2. Check Bearer token if provided
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
        if token == ADMIN_SECRET_KEY:
            admin_user = db.query(User).filter(User.role == "admin").first()
            if admin_user:
                return admin_user
            return User(
                id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                email="admin@betterme.app",
                role="admin",
                full_name="System Admin"
            )
        if not target_id_str:
            target_id_str = token

    if not target_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide X-User-Id or Bearer token."
        )

    try:
        user_uuid = uuid.UUID(target_id_str)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user identification token format."
        )

    user = db.query(User).filter(User.id == user_uuid).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found."
        )

    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: Administrator privileges required."
        )

    return user
