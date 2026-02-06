"""
SQLite database engine and session management.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config import settings
from storage.models import Base

engine = create_engine(settings.SQLITE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Create all tables if they do not exist yet."""
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    """Return a new database session (caller must close)."""
    return SessionLocal()
