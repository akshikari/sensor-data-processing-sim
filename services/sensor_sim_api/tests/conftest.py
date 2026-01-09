import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.main import app
from app.data.sources.db import get_db
from app.data.models.sql import Base, SensorType
from app.core.config import settings


@pytest.fixture(scope="session")
def engine():
    """Create test database engine using PostgreSQL."""
    engine = create_engine(settings.DATABASE_URL, echo=False)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="session")
def SessionLocal(engine: Engine):
    """Create session factory."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db_session(SessionLocal):
    """Create database session with transaction rollback for test isolation."""
    session = SessionLocal()
    session.begin()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def client(db_session: Session):
    """Create test client with overridden database dependency."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def sensor_type(db_session: Session):
    """Create a sensor type for testing."""
    sensor_type = SensorType(name="accelerometer")
    db_session.add(sensor_type)
    db_session.commit()
    db_session.refresh(sensor_type)
    return sensor_type
