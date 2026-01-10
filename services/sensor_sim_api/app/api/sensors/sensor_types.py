import logging
from uuid import UUID
from typing import Annotated

from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.models.api_schemas import (
    SensorTypeCreate,
    SensorTypeUpdate,
    SensorTypeResponse,
    SensorTypeList,
)
from app.domain.sensors import SensorTypeService
from app.core.exceptions import (
    DatabaseError,
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ValidationError,
)
from app.data.sources.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sensor-type", tags=["Sensor Type"])


@router.get(
    "/",
    summary="List sensor types",
    response_model=SensorTypeList,
    response_description="Paginated list of active sensor types",
    responses={
        200: {"description": "List retrieved successfully"},
    },
)
async def get_all_sensor_types(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: int = 0,
    limit: int = 100,
):
    """Retrieve a paginated list of all active sensor types."""
    try:
        service = SensorTypeService(db)
        result = await service.get_all_sensor_types(skip=skip, limit=limit)
        return result
    except DatabaseError as err:
        logger.error("Database error occurred during list sensor types: %s", err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service currently unavailable.",
        ) from err


@router.get(
    "/{id}",
    summary="Get sensor type",
    response_model=SensorTypeResponse,
    response_description="Sensor type details",
    responses={
        200: {"description": "Sensor type retrieved successfully"},
        404: {"description": "Sensor type not found"},
    },
)
async def get_sensor_type(id: UUID, db: Annotated[AsyncSession, Depends(get_db)]):
    """Retrieve the details of a sensor type with the given ID."""
    try:
        service = SensorTypeService(db)
        result = await service.get_sensor_type(id)
        return result
    except ResourceNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(err)
        ) from err
    except DatabaseError as err:
        logger.error(f"Database error occurred: {err}", extra={"recordId": id})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service currently unavailable.",
        ) from err


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    response_model=SensorTypeResponse,
    summary="Create sensor type",
    response_description="Newly created sensor type",
    responses={
        201: {"description": "Sensor type created successfully"},
        409: {"description": "Sensor type already exists"},
        422: {"description": "Sensor type creation failed due to validation errors"},
        500: {
            "description": "Sensor type creation failed due to internal system error"
        },
    },
)
async def create_sensor_type(
    sensor_type: SensorTypeCreate, db: Annotated[AsyncSession, Depends(get_db)]
):
    """Create a new sensor type definition.

    ## Parameters

    - **name** (required): Name of the sensor type (e.g. "accelerometer")
    - **id** (optional): UUID for the sensor type. Auto-generated if omitted.
    """
    try:
        service = SensorTypeService(db)
        result = await service.create_sensor_type(sensor_type)
        return result
    except ResourceAlreadyExistsError as err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(err),
        ) from err
    except ValidationError as err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(err),
        ) from err
    except DatabaseError as err:
        logger.error(
            f"Database error occurred: {err}", extra={"recordId": sensor_type.id}
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service currently unavailable.",
        ) from err


@router.patch(
    "/{id}",
    status_code=status.HTTP_200_OK,
    summary="Update sensor type",
    response_description="Updated sensor type",
    responses={
        200: {"description": "Sensor type updated successfully"},
        404: {"description": "Sensor type not found"},
        422: {"description": "Invalid configuration parameters"},
        503: {"description": "Update failed"},
    },
)
async def update_sensor_type(
    id: UUID,
    sensor_type: SensorTypeUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Update an existing sensor type.

    ## Parameters

    - **id** (required): Identifier for the sensor type
    - **sensor_type** (optional): fields to update
    """
    try:
        service = SensorTypeService(db)
        result = await service.update_sensor_type(id, sensor_type)
        return result
    except ResourceNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(err)
        ) from err
    except ValidationError as err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(err),
        ) from err
    except DatabaseError as err:
        logger.error(f"Database error occurred: {err}", extra={"recordId": id})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service currently unavailable.",
        ) from err


@router.delete(
    "/{id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Archive a sensor type",
    response_description="No content",
    responses={
        204: {"description": "Sensor type successfully archived"},
        404: {"description": "Sensor type not found"},
        500: {"description": "Deletion failed"},
    },
)
async def delete_sensor_type(id: UUID, db: Annotated[AsyncSession, Depends(get_db)]):
    """Archive a sensor type.

    ## Parameters

    - **id** (required): Identifier for the sensor type
    """
    try:
        service = SensorTypeService(db)
        await service.delete_sensor_type(id)
        return
    except ResourceNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sensor type not found.",
        ) from err
    except DatabaseError as err:
        logger.error(f"Database error occurred: {err}", extra={"recordId": id})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Internal server error occurred.",
        ) from err
