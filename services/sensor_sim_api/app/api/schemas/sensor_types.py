"""Pydantic schemas for Sensor Type API endpoints.

This module defines the request and response schemas for the sensor type
CRUD API operations. These schemas handle validation, serialization,
and documentation for the FastAPI endpoints.

The schemas are separated by concern:
    - Create schema: Accept client input for creating resources
    - Update schema: Accept client input for updating resources
    - Response schema: Format API responses with proper JSON serialization
"""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class SensorTypeCreate(BaseModel):
    """Configuration for creating an accelerometer sensor simulator."""

    id: UUID | None = Field(
        default_factory=uuid4,
        description="Sensor type identifier (auto-generated if not provided)",
    )

    name: str = Field(description="Name for the sensor type")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "accelerometer",
            }
        },
    )


class SensorTypeUpdate(BaseModel):
    """Configuration for updating an accelerometer sensor simulator."""

    name: str = Field(description="Name for the sensor type")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "accelerometer",
            }
        },
    )


class SensorTypeResponse(BaseModel):
    """Sensor type id and name"""

    id: UUID = Field(
        description="Sensor type identifier",
    )

    name: str = Field(description="Name for the sensor type")
    create_ts: datetime = Field(
        ...,
        description="Creation timestamp (ISO 8601)",
        examples=["2024-01-15T10:30:00Z"],
    )
    update_ts: datetime | None = Field(
        default=None,
        description="Last update timestamp (ISO 8601)",
        examples=["2024-01-15T10:30:00Z"],
    )
    archived: bool = Field(
        ..., description="Flag marking whether an accelerometer has been deleted or not"
    )
    archive_ts: datetime | None = Field(
        default=None,
        description="Deletion timestamp (ISO 8601)",
        examples=["2024-01-15T10:30:00Z"],
    )

    @field_serializer("id")
    def serialize_uuid(self, value: UUID) -> str:
        """Convert UUID to string for JSON serialization."""
        return str(value)

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "accelerometer",
            }
        },
    )


class SensorTypeList(BaseModel):
    """List of accelerometer sensors with pagination metadata."""

    sensor_types: list[SensorTypeResponse]
    total: int
