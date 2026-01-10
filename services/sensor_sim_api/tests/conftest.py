import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.main import app
from app.data.sources.db import get_db
from app.data.models.sql import Base, SensorType
from app.core.config import settings


@pytest_asyncio.fixture(scope="session")
async def engine():
    """Create test database engine using PostgreSQL."""
    engine = create_async_engine(settings.DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture(scope="session")
def SessionLocal(engine):
    """Create session factory."""
    return async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def db_session(SessionLocal):
    """Create database session with transaction rollback for test isolation."""
    async with SessionLocal() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(db_session):
    """Create test client with overridden database dependency."""

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def sensor_type(db_session: AsyncSession):
    """Create a sensor type for testing."""
    sensor_type = SensorType(name="accelerometer")
    db_session.add(sensor_type)
    await db_session.commit()
    await db_session.refresh(sensor_type)
    return sensor_type
