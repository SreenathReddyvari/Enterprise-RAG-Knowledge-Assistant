"""SQLAlchemy metadata store: document registry + chat history (SQLite by default, Postgres via DATABASE_URL)."""
import datetime as dt

from sqlalchemy import DateTime, Integer, String, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.utils.config import settings

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    category: Mapped[str] = mapped_column(String(100), default="")
    department: Mapped[str] = mapped_column(String(100), default="")
    doc_type: Mapped[str] = mapped_column(String(100), default="")
    access: Mapped[str] = mapped_column(String(50), default="")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(1000))
    answer: Mapped[str] = mapped_column(String(4000))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return SessionLocal()
