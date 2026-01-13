"""Database setup with shared configuration for production and tests."""

from typing import Any, AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def setup_async_db(
    database_url: str,
    echo: bool = False,
    **engine_kwargs: Any,
) -> AsyncEngine:
    """Create SQLAlchemy async engine with configuration.

    :param database_url: Database connection string
    :param echo: Whether to log SQL queries (default: False)
    :param engine_kwargs: Additional arguments passed to create_async_engine

    :return: Configured SQLAlchemy AsyncEngine
    """
    engine = create_async_engine(
        database_url,
        echo=echo,
        **engine_kwargs,
    )

    return engine


engine = setup_async_db(
    settings.DATABASE_URL, echo=(settings.SQLALCHEMY_LOG_LEVEL == "DEBUG")
)
logger.info(f"Database connection established: {settings.DATABASE_URL.split('@')[-1]}")

AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)
logger.info("Database session management setup complete.")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get async database session for FastAPI endpoints.

    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            items = result.scalars().all()
            return items
    """
    async with AsyncSessionLocal() as db:
        yield db
