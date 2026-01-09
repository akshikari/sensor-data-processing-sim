from fastapi import APIRouter

from .sensors import sensors_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(sensors_router)

__all__ = ["api_router"]
