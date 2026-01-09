"""Database setup with shared configuration for production and tests."""

from typing import Any
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def setup_db(
    database_url: str,
    echo: bool = False,
    **engine_kwargs: Any,
) -> Engine:
    """Create SQLAlchemy engine with configuration.

    Args:
        database_url: Database connection string
        echo: Whether to log SQL queries (default: False)
        **engine_kwargs: Additional arguments passed to create_engine

    Returns:
        Configured SQLAlchemy engine
    """
    engine = create_engine(
        database_url,
        echo=echo,
        **engine_kwargs,
    )

    return engine


engine = setup_db(
    settings.DATABASE_URL,
    echo=(settings.SQLALCHEMY_LOG_LEVEL == "DEBUG")
)
logger.info(f"Database connection established: {settings.DATABASE_URL.split('@')[-1]}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
logger.info("Database session management setup complete.")


def get_db():
    """Dependency to get database session for FastAPI endpoints.

    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            items = db.query(Item).all()
            return items
    """
    with SessionLocal() as db:
        yield db
