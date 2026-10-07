import sys
import typing
import typing_extensions

# System patch for stdlib typing generic class lookup on Python 3.11 pre-releases
if hasattr(sys, "modules"):
    sys.modules["typing"]._check_generic = lambda *args, **kwargs: None
    sys.modules["typing_extensions"]._check_generic = lambda *args, **kwargs: None

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


# Enable check_same_thread=False for SQLite local development compatibility
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
