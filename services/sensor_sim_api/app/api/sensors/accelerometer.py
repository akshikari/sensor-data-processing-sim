"""FastAPI routes for accelerometer sensor management.

This module defines the REST API endpoints for managing accelerometer sensors.
It provides CRUD operations and follows RESTful conventions for resource
management.

Endpoints:
    GET /accelerometer/{id} - Retrieve a sensor by ID
    POST /accelerometer - Create a new sensor
    PATCH /accelerometer/{id} - Update an existing sensor
"""

import logging
from uuid import UUID
from typing import Annotated


from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    AccelerometerCreate,
    AccelerometerList,
    AccelerometerUpdate,
    AccelerometerResponse,
)
from app.domain.sensors import AccelerometerService
from app.core.exceptions import (
    DatabaseError,
    InvalidReferenceError,
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ValidationError,
)
from app.data.sources.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/accelerometer", tags=["Accelerometer"])


@router.get(
    "/",
    summary="List accelerometers",
    response_model=AccelerometerList,
    response_description="Paginated list of active accelerometers",
    responses={
        200: {"description": "List retrieved successfully"},
    },
)
async def get_all_accelerometers(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: int = 0,
    limit: int = 100,
):
    """Retrieve a paginated list of all active accelerometer sensors."""
    try:
        service = AccelerometerService(db)
        result = await service.get_all_accelerometers(skip=skip, limit=limit)
        return result
    except DatabaseError as err:
        logger.error("Database error occurred during list accelerometers: %s", err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service currently unavailable.",
        ) from err


@router.get(
    "/{id}",
    summary="Get accelerometer sensor",
    response_model=AccelerometerResponse,
    response_description="Sensor configuration and current state",
    responses={
        200: {"description": "Sensor retrieved successfully"},
        404: {"description": "Sensor not found"},
    },
)
async def get_accelerometer(id: UUID, db: Annotated[AsyncSession, Depends(get_db)]):
    """Retrieve the configuration and current state of an accelerometer sensor with the given ID."""
    try:
        service = AccelerometerService(db)

        result = await service.get_accelerometer(id)

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
    response_model=AccelerometerResponse,
    summary="Create accelerometer sensor",
    response_description="Newly created accelerometer sensor with configuration",
    responses={
        201: {
            "description": "Sensor created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "987fbc97-4bed-5078-9f07-9141ba07c9f3",
                        "sensor_type_id": "123e4567-e89b-12d3-a456-426614174000",
                        "generate_data_params": {
                            "gait_frequency_hz": 2.0,
                            "amplitude_sway_m": 0.05,
                            "amplitude_bounce_m": 0.02,
                        },
                        "anomalous_data_params": {
                            "z_amp_modifier": 0.8,
                            "step_frequency": 4,
                        },
                        "stream_state": None,
                        "create_ts": "2024-01-15T10:30:00Z",
                        "update_ts": "2024-01-15T10:30:00Z",
                    }
                }
            },
        },
        422: {
            "description": "Accelerometer creation failed due to data constraint violations."
        },
        500: {
            "description": "Accelerometer creation failed due to internal system error."
        },
    },
)
async def create_accelerometer(
    accelerometer: AccelerometerCreate, db: Annotated[AsyncSession, Depends(get_db)]
):
    """Create a virtual accelerometer sensor that generates realistic gait motion data.

    Configure a sensor to simulate normal walking patterns with optional anomalous behavior.
    Each sensor generates continuous 3-axis accelerometer data based on the specified parameters.

    ## Parameters

    - **sensor_type_id** (required): Identifier for the sensor type
    - **generate_data_params** (optional): Normal gait simulation configuration
      - Walking frequency, sway/bounce amplitudes, noise levels
      - Defaults provided for typical human walking patterns
    - **anomalous_data_params** (optional): Anomaly simulation configuration
      - Simulate irregular gait patterns, reduced vertical motion, etc.

    ## Example Request
    ```json
    {
        "sensor_type_id": "123e4567-e89b-12d3-a456-426614174000",
        "generate_data_params": {
            "gait_frequency_hz": 2.0,
            "amplitude_sway_m": 0.05
        },
        "anomalous_data_params": {
            "z_amp_modifier": 0.8,
            "step_frequency": 4
        }
    }
    ```
    """
    try:
        service = AccelerometerService(db)
        result = await service.create_accelerometer(accelerometer)
        return result
    except ResourceAlreadyExistsError as err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(err),
        ) from err
    except (ValidationError, InvalidReferenceError) as err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(err),
        ) from err
    except DatabaseError as err:
        logger.error(
            f"Database error occurred: {err}", extra={"recordId": accelerometer.id}
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service currently unavailable.",
        ) from err


@router.patch(
    "/{id}",
    status_code=status.HTTP_200_OK,
    summary="Update accelerometer sensor",
    response_description="Updated sensor configuration",
    responses={
        200: {"description": "Sensor updated successfully"},
        404: {"description": "Sensor not found"},
        422: {"description": "Invalid configuration parameters"},
        503: {"description": "Update failed"},
    },
)
async def update_accelerometer(
    id: UUID,
    accelerometer: AccelerometerUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Update the configuration of an existing accelerometer sensor.

    Modify data generation parameters, anomaly settings, or streaming state.
    Only provide the fields you want to change - all fields are optional.

    ## Parameters

    - **id** (required): Identifier for the accelerometer
    - **accelerometer** (optional): fields to update with their new values

    ## Example Request
    ```json
    {
        "generate_data_params": {
            "gait_frequency_hz": 2.5
        }
    }
    ```
    """
    try:
        service = AccelerometerService(db)
        result = await service.update_accelerometer(id, accelerometer)
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
    summary="Mark an accelerometer as archived.",
    response_description="No content",
    responses={
        204: {"description": "Accelerometer successfully archived"},
        404: {"description": "Accelerometer with given ID not found"},
        500: {"description": "Deletion failed"},
    },
)
async def delete_accelerometer(id: UUID, db: Annotated[AsyncSession, Depends(get_db)]):
    """Delete an accelerometer with the given ID.

    ## Parameters

    - **id** (required): Identifier for the accelerometer
    """
    try:
        service = AccelerometerService(db)
        await service.delete_accelerometer(id)
        return
    except ResourceNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No accelerometer found with given ID.",
        ) from err
    except DatabaseError as err:
        logger.error(f"Database error occurred: {err}", extra={"recordId": id})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Internal server error occurred.",
        ) from err
