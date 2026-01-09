"""Application configuration using Pydantic Settings."""

import os
from functools import lru_cache
from typing import Literal

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Environment
    ENV: Literal["dev", "test", "staging", "prod"] = "dev"

    # PostgreSQL Database Components
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "sensor_sim_dev"
    POSTGRES_USER: str = "user"
    POSTGRES_PASSWORD: str = "password"

    # Logging
    LOG_LEVEL: str = "INFO"
    SQLALCHEMY_LOG_LEVEL: str = "WARNING"

    # API
    API_TITLE: str = "Robotics IMU Sensor Data API"
    API_VERSION: str = "0.1.0"

    # Security
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        """Construct DATABASE_URL from PostgreSQL components."""
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    model_config = SettingsConfigDict(
        env_file=f".env.{os.getenv('ENV', 'dev')}",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
