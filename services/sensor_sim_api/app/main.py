from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.routing import APIRoute

from app.api import api_router
from app.core.logging import setup_logging, get_logger
from app.core.config import settings

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
setup_logging(level=LOG_LEVEL)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    yield


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


tags_metadata = [
    {
        "name": "Accelerometer",
        "description": "Manage accelerometer sensors that simulate realistic gait motion data. "
        "Create, configure, and control virtual accelerometers for testing and simulation.",
    },
]

app = FastAPI(
    title="Robotics IMU Sensor Data API",
    description="REST API for simulating IMU (Inertial Measurement Unit) sensor data for robotics applications. "
    "Generate realistic accelerometer and gyroscope data with configurable parameters and anomaly patterns.",
    version="0.1.0",
    openapi_tags=tags_metadata,
    generate_unique_id_function=custom_generate_unique_id,
    lifespan=lifespan,
)

app.include_router(api_router)

# TODO: CORS


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
    }
