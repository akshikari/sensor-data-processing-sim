"""Pydantic schemas for Accelerometer API endpoints.

This module defines the request and response schemas for the accelerometer
sensor CRUD API operations. These schemas handle validation, serialization,
and documentation for the FastAPI endpoints.

The schemas are separated by concern:
    - Create schema: Accept client input for creating resources
    - Update schema: Accept client input for updating resources
    - Response schema: Format API responses with proper JSON serialization
"""

from datetime import datetime
import math
from uuid import UUID

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


def _norm_phase(radians: float) -> float:
    """Normalize angle to [0, 2pi]"""
    two_pi = math.tau
    r = radians % two_pi
    return r if r >= 0 else r + two_pi


class GenerateDataParamsCreate(BaseModel):
    """Data model for accelerometer-specific arguments for generating data"""

    # Motion Parameters
    gait_frequency_hz: float = Field(
        default=2.0,
        gt=0,
        description="The stepping cadence in steps per second (Hz). Controls the primary frequency of the gait cycle.",
    )

    # Oscillation Amplitudes (Spatial)
    amplitude_sway_m: float = Field(
        default=0.05,
        gt=0,
        description="Peak magnitude of lateral (side-to-side) displacement in meters.",
    )
    amplitude_bounce_m: float = Field(
        default=0.02,
        ge=0,
        description="Peak magnitude of vertical (up-and-down) displacement in meters.",
    )

    # Oscillation Amplitudes (Rotational)
    amplitude_roll_rad: float = Field(
        default=0.05,
        ge=0,
        description="Peak magnitude of rotational oscillation around the roll axis (side-to-side tilt) in radians.",
    )
    amplitude_pitch_rad: float = Field(
        default=0.05,
        ge=0,
        description="Peak magnitude of rotational oscillation around the pitch axis (forward-backward tilt) in radians.",
    )

    # Base Orientation (Static Offsets)
    base_pitch_rad: float = Field(
        default=0.0,
        description="Static pitch offset in radians. Simulates a permanent forward/backward lean or sensor mounting angle.",
    )
    base_roll_rad: float = Field(
        default=0.03,
        description="Static roll offset in radians. Simulates a permanent side tilt or sensor mounting asymmetry.",
    )

    # Phase Shifts (Timing/Synchronization)
    phase_sway_rad: float = Field(
        default=0.0,
        description="Phase offset for lateral sway in radians. Adjusts when the sway peak occurs relative to the gait cycle.",
    )
    phase_bounce_rad: float = Field(
        default=np.pi / 2,
        description="Phase offset for vertical bounce in radians. Adjusts the timing of the peak height relative to the step.",
    )
    phase_roll_rad: float = Field(
        default=np.pi, description="Phase offset for roll oscillation in radians."
    )
    phase_pitch_rad: float = Field(
        default=0.0, description="Phase offset for pitch oscillation in radians."
    )

    # Sensor Characteristics
    noise_std_dev: float = Field(
        default=0.05,
        ge=0,
        description="Standard deviation of the Gaussian noise added to the signal to simulate sensor imperfections.",
    )

    # Physics
    gravity_mps2: float = Field(
        default=9.81,
        gt=0,
        description="Gravitational acceleration constant in m/s² (Earth standard is approx 9.81).",
    )

    @field_validator(
        "phase_sway_rad",
        "phase_bounce_rad",
        "phase_roll_rad",
        "phase_pitch_rad",
    )
    @classmethod
    def apply_phase_normalization(cls, v: float) -> float:
        return _norm_phase(v)


class AnomalousDataModifierParamsCreate(BaseModel):
    """Data model for parameters to add anomalous data to the generated accelerometer data"""

    z_amp_modifier: float | None = Field(
        default=None,
        gt=0,
        description="Simulate a leg having a weaker push off the ground with each step",
    )

    time_drift_offset: float | None = Field(
        default=None, ge=0, description="Simulate sensor error and time drift"
    )

    step_time_delay: float | None = Field(
        default=None,
        ge=0,
        description="Simulate a delay caused by slower motion of a leg with each step",
    )

    step_frequency: int = Field(
        default=4,
        gt=0,
        description="On which step the anomalous modifier(s) should apply. Default every 4th step",
    )

    @model_validator(mode="after")
    def require_frequency_if_modifier_set(self):
        modifiers = [self.z_amp_modifier, self.time_drift_offset, self.step_time_delay]

        is_modifying = any(m is not None for m in modifiers)

        if is_modifying and self.step_frequency is None:
            raise ValueError(
                "If any anomalous modifier (z_amp, time_drift, delay) is provided, "
                "step_frequency must also be specified."
            )
        return self


class AnomalyState(BaseModel):
    """State of persisted anomalous data such as time drift or step delay."""

    cumulative_time_drift: float | None = Field(default=None)
    cumulative_step_delay: float | None = Field(default=None)
    last_step_idx_seen: int | None = Field(default=None)


class StreamState(BaseModel):
    """
    State of persisted stream metadata.
    We will derive the local monotonic baseline from 'start_ts_utc' at runtime.
    """

    id: UUID = Field(..., description="Unique identifier for the stream/sensor.")
    start_ts_utc: datetime = Field(
        ...,
        description="UTC timestamp when the stream effectively started. Anchors the physics simulation.",
    )
    anomaly_state: AnomalyState = Field(
        ...,
        description="Current state of any applied anomalies.",
    )
    sample_index: int = Field(
        default=0, ge=0, description="The current sequence number."
    )


class AccelerometerCreate(BaseModel):
    """Configuration for creating an accelerometer sensor simulator."""

    id: UUID | None = Field(
        default=None, description="Sensor identifier (auto-generated if not provided)"
    )
    sensor_type_id: UUID = Field(
        ...,
        description="Sensor type identifier",
        examples=["123e4567-e89b-12d3-a456-426614174000"],
    )
    generate_data_params: GenerateDataParamsCreate = Field(
        default_factory=lambda: GenerateDataParamsCreate(),
        description="Normal gait simulation parameters (uses defaults if not specified)",
    )
    anomalous_data_params: AnomalousDataModifierParamsCreate | None = Field(
        default=None,
        description="Anomaly simulation parameters (omit for normal gait only)",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "sensor_type_id": "123e4567-e89b-12d3-a456-426614174000",
                "generate_data_params": {
                    "gait_frequency_hz": 2.0,
                    "amplitude_sway_m": 0.05,
                },
                "anomalous_data_params": {
                    "z_amp_modifier": 0.8,
                    "step_frequency": 4,
                },
            }
        },
    )


class GenerateDataParamsUpdate(BaseModel):
    """
    Partial update schema.
    Notice we re-use the Field constraints (gt=0) so invalid updates are rejected.
    """

    gait_frequency_hz: float | None = Field(default=None, gt=0)
    amplitude_sway_m: float | None = Field(default=None, gt=0)
    amplitude_bounce_m: float | None = Field(default=None, ge=0)
    amplitude_roll_rad: float | None = Field(default=None, ge=0)
    amplitude_pitch_rad: float | None = Field(default=None, ge=0)

    base_pitch_rad: float | None = None
    base_roll_rad: float | None = None

    phase_sway_rad: float | None = None
    phase_bounce_rad: float | None = None
    phase_roll_rad: float | None = None
    phase_pitch_rad: float | None = None

    noise_std_dev: float | None = Field(default=None, ge=0)
    gravity_mps2: float | None = Field(default=None, gt=0)

    @field_validator(
        "phase_sway_rad",
        "phase_bounce_rad",
        "phase_roll_rad",
        "phase_pitch_rad",
        mode="after",
    )
    @classmethod
    def apply_phase_normalization(cls, v: float | None) -> float | None:
        if v is None:
            return None
        return _norm_phase(v)


class AnomalousDataModifierParamsUpdate(BaseModel):
    """
    Partial update schema with Conditional Logic.
    """

    z_amp_modifier: float | None = Field(default=None, gt=0)
    time_drift_offset: float | None = Field(default=None, ge=0)
    step_time_delay: float | None = Field(default=None, ge=0)
    step_frequency: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def require_frequency_if_modifier_set(self):
        modifiers = [self.z_amp_modifier, self.time_drift_offset, self.step_time_delay]

        is_modifying = any(m is not None for m in modifiers)

        if is_modifying and self.step_frequency is None:
            raise ValueError(
                "If any anomalous modifier (z_amp, time_drift, delay) is provided, "
                "step_frequency must also be specified."
            )
        return self


class AccelerometerUpdate(BaseModel):
    """Update configuration for an existing accelerometer sensor.

    All fields are optional - only specify the parameters you want to change
    """

    generate_data_params: GenerateDataParamsUpdate | None = None
    anomalous_data_params: AnomalousDataModifierParamsUpdate | None = None
    stream_state: StreamState | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AccelerometerResponse(BaseModel):
    """Accelerometer sensor configuration and metadata."""

    id: UUID = Field(
        ...,
        description="Sensor identifier",
        examples=["987fbc97-4bed-5078-9f07-9141ba07c9f3"],
    )
    sensor_type_id: UUID = Field(
        ...,
        description="Sensor type identifier",
        examples=["123e4567-e89b-12d3-a456-426614174000"],
    )
    generate_data_params: GenerateDataParamsCreate = Field(
        ..., description="Normal gait simulation parameters"
    )
    anomalous_data_params: AnomalousDataModifierParamsCreate | None = Field(
        default=None, description="Anomaly simulation parameters (null if disabled)"
    )
    stream_state: StreamState | None = Field(
        default=None, description="Current streaming state (null if inactive)"
    )
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

    @field_serializer("id", "sensor_type_id")
    def serialize_uuid(self, value: UUID) -> str:
        """Convert UUID to string for JSON serialization."""
        return str(value)

    @field_serializer("generate_data_params", "anomalous_data_params", "stream_state")
    def serialize_dataclass(
        self,
        value: GenerateDataParamsCreate
        | AnomalousDataModifierParamsUpdate
        | StreamState
        | None,
    ):
        """Convert configuration objects to dictionaries for JSON serialization."""
        if value is None:
            return None
        return value.model_dump(mode="python")

    model_config = ConfigDict(
        from_attributes=True,  # Allows SQLAlchemy model → Pydantic
        arbitrary_types_allowed=True,  # Allow dataclasses
    )


class AccelerometerList(BaseModel):
    """List of accelerometer sensors with pagination metadata."""

    accelerometers: list[AccelerometerResponse]
    total: int
