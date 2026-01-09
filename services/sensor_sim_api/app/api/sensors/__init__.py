from fastapi import APIRouter
from .accelerometer import router as accelerometer_router
from .sensor_types import router as sensor_type_router

sensors_router = APIRouter(prefix="/sensors")

sensors_router.include_router(accelerometer_router)
sensors_router.include_router(sensor_type_router)

__all__ = ["sensors_router"]
