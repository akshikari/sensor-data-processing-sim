"""FastAPI routes for sensor type management.

This module defines the REST API endpoints for managing sensor type.
Endpoints:
    GET /sensor_type/{id} - Retrieve a sensor by ID
    POST /sensor_type - Create a new sensor
    PATCH /sensor_type/{id} - Update an existing sensor
    DELETE /sensor_type/{id} - Soft delete an existing sensor type
"""

import logging
from uuid import UUID
from typing import Annotated


from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session

from app.data.models.api_schemas import (
    SensorTypeCreate,
    SensorTypeUpdate,
    SensorTypeResponse,
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
    "/{id}",
    summary="Get sensor type",
    response_model=SensorTypeResponse,
    response_description="Sensor configuration and current state",
    responses={
        200: {"description": "Sensor retrieved successfully"},
        404: {"description": "Sensor not found"},
    },
)
def get_sensor_type(id: UUID, db: Annotated[Session, Depends(get_db)]):
    """Retrieve the configuration and current state of an sensor type with the given ID."""
    try:
        service = SensorTypeService(db)

        result = service.get_sensor_type(id)

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
    response_description="Newly created sensor type.",
    responses={
        201: {
            "description": "Sensor created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "name": "accelerometer",
                    }
                }
            },
        },
        422: {
            "description": "Sensor type creation failed due to data constraint violations."
        },
        500: {
            "description": "Sensor type creation failed due to internal system error."
        },
    },
)
def create_sensor_type(
    sensor_type: SensorTypeCreate, db: Annotated[Session, Depends(get_db)]
):
    """Create a virtual sensor type

    ## Parameters

    - **id** (required): Identifier for the sensor type
    - **name** (required): Name of the sensor type (accelerometer, gyroscope, etc.)

    ## Example Request
    ```json
    {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "name": "accelerometer"
    }
    ```
    """
    try:
        service = SensorTypeService(db)
        result = service.create_sensor_type(sensor_type)
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
def update_sensor_type(
    id: UUID,
    sensor_type: SensorTypeUpdate,
    db: Annotated[Session, Depends(get_db)],
):
    """Update an existing sensor type.

    ## Parameters

    - **sensor_type** (required): fields to update with their new values

    ## Example Request
    ```json
    {
        "name": "gyroscope"
    }
    ```
    """
    try:
        service = SensorTypeService(db)
        result = service.update_sensor_type(id, sensor_type)
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
    summary="Mark a sensor type as archived.",
    response_description="No content",
    responses={
        204: {"description": "Sensor type successfully archived"},
        404: {"description": "Sensor type with given ID not found"},
        500: {"description": "Deletion failed"},
    },
)
def delete_sensor_type(id: UUID, db: Annotated[Session, Depends(get_db)]):
    """Delete a sensor type with the given ID.

    ## Parameters

    - **id** (required): Identifier for the sensor type
    """
    try:
        service = SensorTypeService(db)
        service.delete_sensor_type(id)
        return
    except ResourceNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No sensor type found with given ID.",
        ) from err
    except DatabaseError as err:
        logger.error(f"Database error occurred: {err}", extra={"recordId": id})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Internal server error occurred.",
        ) from err
